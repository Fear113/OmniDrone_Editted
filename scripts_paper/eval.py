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

import imageio

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

transport_checkpoint = None
formation_checkpoint = None

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless)
    agent_spec: AgentSpec = env.agent_spec["drone"]
    transport_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")
    formation_policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")

    if transport_checkpoint is not None:
        transport_policy.load_state_dict(torch.load(transport_checkpoint))
    if formation_checkpoint is not None:
        formation_policy.load_state_dict(torch.load(formation_checkpoint))

    def record_frame(frames, *args, **kwargs):
        frame = env.render(mode="rgb_array")
        frames.append(frame)

    frames = []
    seed = 1
    num_payloads = cfg.task.num_payloads_per_group

    env.enable_render(True)
    env.set_seed(seed)
    state = env.reset()

    for i in range(num_payloads):
        # formation
        while not state['done']:
            state = env.step(formation_policy(state, deterministic=True))['next']
            record_frame(frames)

        with torch.no_grad():
            state_snapshot = env.snapshot_state()

        simulation_app.context.close_stage()
        simulation_app.context.new_stage()

        env = env_class(cfg, headless=cfg.headless, initial_state=state_snapshot)   
        state = env.reset()

        # transport
        while not state['done']:
            state = env.step(transport_policy(state, deterministic=True))['next']
            record_frame(frames)

        with torch.no_grad():
            state_snapshot = env.snapshot_state()

        simulation_app.context.close_stage()
        simulation_app.context.new_stage()

        env = env_class(cfg, headless=cfg.headless, initial_state=state_snapshot)  
        state = env.reset()

    if len(frames):
        imageio.mimsave("result_video/video.mp4", frames, fps=0.5 / cfg.sim.dt)
        print("completed the video")

if __name__ == "__main__":
    main()
