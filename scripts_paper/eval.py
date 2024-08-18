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
formation_checkpoint = "./formation_checkpoint.pt"
# "./formation_checkpoint.pt"

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

    state = env.reset()
    while True:
        while not state['done']:
            random_action = env.rand_action(state)
            state = env.step(random_action)['next']
            # state = env.step(transport_policy(state, deterministic=True))['next']

        state_snapshot = env.snapshot_state()
        simulation_app.context.close_stage()
        simulation_app.context.new_stage()
        env = env_class(cfg, headless=cfg.headless, initial_state=state_snapshot)
        state = env.reset()


if __name__ == "__main__":
    main()
    main()
