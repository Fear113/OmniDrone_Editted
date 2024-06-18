import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
import torch
import imageio


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    env = env_class(cfg, headless=False)
    env.set_seed(cfg.seed)
    envs = torch.arange(cfg.env.num_envs)
    frames = []

    init_pos = env.group.get_world_poses()[0]
    init_rot = env.group.get_world_poses()[1]
    init_joint_pos = env.group.get_joint_positions()
    target_pos = env.payload_target_pos

    steps = 75
    i = 0
    middle_target = torch.tensor([[[-1.0, 0, 0.6], [-4.5, 0, 1], [-1.0, 0, 2.7]]], device=init_pos.device)
    add = (middle_target - init_pos) / steps

    while i < steps:
        env.rand_step()
        env.group.set_world_poses(init_pos + add * i, init_rot, envs)
        env.group.set_joint_positions(init_joint_pos, envs)
        frames.append(env.render(mode="rgb_array"))
        i += 1

    i = 0
    add = (target_pos - middle_target) / steps

    while i < steps:
        env.rand_step()
        env.group.set_world_poses(middle_target + add * i, init_rot, envs)
        env.group.set_joint_positions(init_joint_pos, envs)
        frames.append(env.render(mode="rgb_array"))
        i += 1

    imageio.mimsave("video.mp4", frames, fps=15)

    simulation_app.close()

if __name__ == "__main__":
    main()
