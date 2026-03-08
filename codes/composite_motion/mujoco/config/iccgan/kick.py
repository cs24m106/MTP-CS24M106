env_cls = "ICCGANHumanoid"
env_params = dict(
    motion_file = "assets/motions/iccgan/kick.json"
)

training_params = dict(
    max_epochs = 10000,
    save_interval = 500,
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}
