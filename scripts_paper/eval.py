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

# modify - 1
class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1
import imageio
# modify - 1

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

# modify - 2
    def record_frame(frames, *args, **kwargs):
            frame = env.render(mode="rgb_array")
            frames.append(frame)

    frames_transport = []
    frames_formation = []
    seed = 1

    env.enable_render(True)
    env.eval()
    env.set_seed(seed)
    state = env.reset()

    print("start formation")

    while not state['done']:
        state = env.step(formation_policy(state, deterministic=True))['next']
        record_frame(frames_formation)

    print("finish formation")

    with torch.no_grad():
        state_snapshot = env.snapshot_state()

    simulation_app.context.close_stage()
    simulation_app.context.new_stage()

    env = env_class(cfg, headless=cfg.headless, initial_state=state_snapshot)  
    state = env.reset()

    print("start transport")

    while not state['done']:
        state = env.step(transport_policy(state, deterministic=True))['next']
        record_frame(frames_transport)

    print("finish transport")

    if len(frames_formation):
        imageio.mimsave("result_video/formation_video.mp4", frames_formation, fps=0.5 / cfg.sim.dt)
        print("completed the formation video")
    if len(frames_transport):
        imageio.mimsave("result_video/transport_video.mp4", frames_transport, fps=0.5 / cfg.sim.dt)
        print("completed the transport video")
    
    frames_total = frames_formation + frames_formation

    if len(frames_total):
        imageio.mimsave("result_video/total_video.mp4", frames_total, fps=0.5 / cfg.sim.dt)
        print("completed the total video")

    print("done everything")
# modify - 2

if __name__ == "__main__":
    main()
    main()
