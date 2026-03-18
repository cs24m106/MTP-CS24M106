env_cls = "ICCGANHumanoid"
env_params = dict(
    max_cycles = 2,         # slow & steady paced, no jump involved, seamlessly loopable
    loop_phase_obs = False,  # preferable to use this on loopable motions to converge faster
    motion_file = "assets/motions/iccgan/limp_walk.json"
)

training_params = dict(
    max_epochs = 10000,
    save_interval = 500,
    terminate_reward = -1,
    sym_loss_coeff = 0.00,
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}
