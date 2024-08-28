from typing import Optional

import hydra
from omegaconf import OmegaConf
import torch
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.learning import (
    MAPPOPolicy, 
    HAPPOPolicy,
    QMIXPolicy,
    DQNPolicy,
    SACPolicy,
    TD3Policy,
    MATD3Policy,
    TDMPCPolicy,
)
from torchrl.envs.transforms import (
    TransformedEnv,
    InitTracker,
    Compose,
)
import imageio
from omni_drones.utils.torchrl.transforms import (
    ravel_composite,
)

algos = {
    "mappo": MAPPOPolicy,
    "happo": HAPPOPolicy,
    "qmix": QMIXPolicy,
    "dqn": DQNPolicy,
    "sac": SACPolicy,
    "td3": TD3Policy,
    "matd3": MATD3Policy,
    "tdmpc": TDMPCPolicy,
}

transport_checkpoint = "scripts_paper/transport_checkpoint.pt"
formation_checkpoint = "scripts_paper/formation_checkpoint.pt"
# "./formation_checkpoint.pt"

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.envs.logistics.state_snapshot import StateSnapshot

    def get_env(name, config_path, headless, initial_state: Optional[StateSnapshot] = None):
        cfg = hydra.compose(config_name="train", overrides=[f"task={config_path}", "task.env.num_envs=1"])
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        if initial_state is None:
            env = IsaacEnv.REGISTRY[name](cfg, headless=headless).eval()
        else:
            env = IsaacEnv.REGISTRY[name](cfg, headless=headless, initial_state=initial_state).eval()

        transforms = [InitTracker()]
        if cfg.task.get("ravel_obs", False):
            transform = ravel_composite(env.observation_spec, ("agents", "observation"))
            transforms.append(transform)
        transforms = Compose(*transforms)
        env = TransformedEnv(env, transforms).eval()

        env.set_seed(cfg.seed)
        env.reset()

        return env, transforms

    transport_env, transport_transform = get_env(name='TransportHover', config_path='Transport/TransportHover', headless=True)
    transport_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=transport_env.agent_spec["drone"],
                                                    device="cuda")
    transport_policy.load_state_dict(torch.load(transport_checkpoint))
    simulation_app.context.close_stage()
    simulation_app.context.new_stage()

    formation_env, formation_transform = get_env(name='Formation', config_path='Formation', headless=True)
    formation_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=formation_env.agent_spec["drone"],
                                                    device="cuda")
    formation_policy.load_state_dict(torch.load(formation_checkpoint))
    simulation_app.context.close_stage()
    simulation_app.context.new_stage()

    env, _ = get_env(name='Logistics', config_path='Logistics', headless=cfg.headless)

    frames = []
    seed = 1

    env.enable_render(True)
    env.set_seed(seed)
    td = env.reset()

    state_snapshot = env.snapshot_state()
    max_tasks = 10

    for i in range(max_tasks):
        while not td['done']:
            actions = []

            for j, group in enumerate(state_snapshot.group_snapshots):
                if group.is_transporting:
                    transformed_state = transport_transform._step(env.get_transport_state(j), env.get_transport_state(j))
                    actions.append(transport_policy(transformed_state, deterministic=True)['agents']['action'])
                else:
                    transformed_state = formation_transform._step(env.get_formation_state(j), env.get_formation_state(j))
                    actions.append(formation_policy(transformed_state, deterministic=True)['agents']['action'])

            env.rulebased_collison_avoidance()

            td['agents']['action'] = torch.cat(actions, dim=1)
            td = env.step(td)['next']
            record_frame(frames, env)

        with torch.no_grad():
            state_snapshot = env.snapshot_state()
        simulation_app.context.close_stage()
        simulation_app.context.new_stage()

        env, _ = get_env(name='Logistics', config_path='Logistics', headless=cfg.headless, initial_state=state_snapshot)
        td = env.reset()

    if len(frames):
        imageio.mimsave("video.mp4", frames, fps=0.5 / cfg.sim.dt)
        print("completed the video")


def record_frame(frames, env, *args, **kwargs):
    frame = env.render(mode="rgb_array")
    frames.append(frame)

if __name__ == "__main__":
    main()
