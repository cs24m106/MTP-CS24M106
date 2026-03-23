env_cls = "ICCGANHumanoidTarget"
env_params = dict(
    motion_file = "assets/motions/clips_walk.yaml",
    goal_reward_weight = [0.5],

    goal_radius = 0.5,
    sp_lower_bound = 1.2,           # Minimum speed for reward calculation
    sp_upper_bound = 1.5,           # Maximum speed for reward calculation
    goal_timer_range = (90, 150),
    goal_sp_mean = 1,               # Target speed for navigation
    goal_sp_std = 0.25,             # Speed variation standard deviation
    goal_sp_min = 0,                # Speed min clipping bound
    goal_sp_max = 1.25              # Speed max clipping bound
)

training_params = dict(
    max_epochs = 50000,
    save_interval = 500,
    terminate_reward = -25
)

discriminators = {
    "walk/full": dict(
        parent_link = None,
    )
}
