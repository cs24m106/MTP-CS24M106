env_cls = "ICCGANHumanoid"
env_params = dict(
    episode_length = 300,
    motion_file = "assets/motions/iccgan/limp_walk.json"
)

training_params = dict(
    max_epochs = 5000,
    save_interval = 250,
    terminate_reward = -1,
    loop_phase_obs = True,
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}
