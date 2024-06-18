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

    init_pos = env.drone.get_world_poses()[0]
    init_rot = env.drone.get_world_poses()[1]
    target_pos = init_pos.clone()
    target_pos[..., 0] += 5

    steps = 150
    i = 0
    add = (target_pos - init_pos) / steps

    while i < steps:
        env.rand_step()
        env.drone.set_world_poses(init_pos + add * i, init_rot, envs)
        frames.append(env.render(mode="rgb_array"))
        i += 1

    imageio.mimsave("video.mp4", frames, fps=15)
    simulation_app.close()


if __name__ == "__main__":
    main()
