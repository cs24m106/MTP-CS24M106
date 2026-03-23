env_cls = "ICCGANHumanoid"
env_params = dict(
    motion_file = "assets/motions/iccgan/roll.json",
    grace_steps = 30, # ground contact time at least 1s i.e. 30 steps for 30fps (for larger duration slow increase over time)
)

training_params = dict(
    max_epochs = 10000,
    save_interval = 500,
    terminate_reward = -1
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}
