env_cls = "ICCGANHumanoid"
env_params = dict(
    motion_file = "assets/motions/iccgan/spinning_jump.json"
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
