# ICCGAN: Joyful Walk
# Style: Happy, energetic walk with arm swings

env_cls = "ICCGANHumanoidMujoco"

env_params = dict(
    episode_length=300,
    motion_file="assets/motions/iccgan/joyful_walk.json",
    character_model="assets/humanoid.xml",
)

training_params = dict(
    max_epochs=10000,
    save_interval=2000,
    terminate_reward=-1,
    num_envs=32,
    batch_size=64,
)

discriminators = {
    "_/full": dict(
        parent_link=None,
    )
}
