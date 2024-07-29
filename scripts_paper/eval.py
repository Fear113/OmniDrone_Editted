import hydra
from omegaconf import OmegaConf
import torch
from omni_drones import CONFIG_PATH, init_simulation_app


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.envs.logistics.utils import InitialState, DisconnectedPayload, Group
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    initial_state = InitialState([Group(None, None, None, False, [DisconnectedPayload(None, None)])])
    env = env_class(cfg, headless=cfg.headless, initial_state=initial_state)
    state = env.reset()
    i = 0

    while True:
        state = env.step(env.rand_action(state))
        i += 1
        # if i == 10:
        #     base_env.reset()
        # action = base_env.rand_action(state)['agents']['action']
        # action = torch.full(action.shape, 0)
        # action[...,0] = 1
        # action[...,1] = 1
        # state['agents']['action'] = action
        # state = base_env.step(state)


if __name__ == "__main__":
    main()
