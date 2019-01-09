from gym.envs.registration import registry, register, make, spec
# import sys
# sys.path.insert(0, '/ddpg')
# sys.path.insert(0, '/ddpg/nets')

register(
    id='quadreped-robot-v0',
    entry_point='envs.quadreped:QuadrepedEnv',
    max_episode_steps=10000,
    reward_threshold=4800.0,
)