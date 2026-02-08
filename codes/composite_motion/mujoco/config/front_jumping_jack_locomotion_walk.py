# Composite Motion: Jumping Jack + Walking
# Arm movements from jumping jack while walking

env_cls = "ICCGANHumanoidMujoco"

env_params = dict(
    episode_length=500,
    motion_file="assets/motions/iccgan/front_jumping_jack_locomotion_walk.json",
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
    "jumping_jack/upper": dict(
        key_links=["torso", "head", "right_upper_arm", "right_lower_arm", "right_hand", 
                   "left_upper_arm", "left_lower_arm", "left_hand"],
        parent_link="pelvis",
        weight=0.5,
    ),
    "walk/lower": dict(
        key_links=["right_thigh", "right_shin", "right_foot", 
                   "left_thigh", "left_shin", "left_foot"],
        parent_link="pelvis",
        weight=0.5,
    )
}
