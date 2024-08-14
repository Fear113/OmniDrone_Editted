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


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.envs.logistics.utils import InitialState, DisconnectedPayload, Group, ConnectedPayload
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    initial_state = InitialState([Group(None, None, None, False, [DisconnectedPayload(None, None)])])
    env = env_class(cfg, headless=cfg.headless, initial_state=initial_state)
    state = env.reset()
    i = 0

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
    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")
    state_dict = torch.load('./checkpoint_28809600.pt')
    # frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    # total_frames = 1
    # collector = SyncDataCollector(
    #     env,
    #     policy=policy,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     device=cfg.sim.device,
    #     return_same_td=True,
    # )
    policy.load_state_dict(state_dict)
    while True:
        # random_input = env.rand_action(state)
        # action = env.rand_action(state)['agents']['action']
        # action = torch.full(action.shape, -1)
        # state['agents']['action'] = action
        # state = env.step(state)['next']
        # state = env.step(env.rand_action(state)) ['agents']['action']
        state = env.step(policy(state, deterministic=True))['next']
        i += 1
        # if i == 10:
        #     base_env.reset()
        # action = base_env.rand_action(state)['agents']['action']
        # action = torch.full(action.shape, 0)
        # action[...,0] = 1
        # action[...,1] = 1
        # state['agents']['action'] = action
        # state = base_env.step(state)
        ###할것 : done 확인하고 끝내기.
        ###state 얻기. 
        if state['done']:
            break
    cPos = state['agents']['observation_central']['drones'].cpu().detach().numpy()[:,:,:3]
    cRot = state['agents']['observation_central']['drones'].cpu().detach().numpy()[:,:,3:7]
    cVel = state['agents']['observation_central']['drones'].cpu().detach().numpy()[:,:,7:13]
    initial_state = InitialState([Group(cPos, cRot,\
                                         cVel, True, [ConnectedPayload])])
    ### pos-3, rot-4, vel-6, heading-3, up-3, throttle-4
    simulation_app.context.close_stage()
    simulation_app.context.new_stage()
    # simulation_app.context.new_stage_async()
    # simulation_app.context.reset()
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=cfg.headless, initial_state=initial_state)
    state = env.reset()
    while True:
        simulation_app.update()

    simulation_app.close()
    while True:
        state = env.step(env.rand_action(state))


if __name__ == "__main__":
    main()
    main()
