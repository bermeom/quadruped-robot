from sources.source_gym import source_gym
import numpy as np
import gym

gym.envs.register(
    id='quadreped-robot-v0',
    entry_point='envs.quadreped:QuadrepedEnv',
    max_episode_steps=10000,
    reward_threshold=4800.0,
)


##### SOURCE GYM HALFCHEETAH
class source_quadreped_robot( source_gym ):

    ### __INIT__
    def __init__( self ):

        source_gym.__init__( self , 'quadreped-robot-v0' )

    ### INFORMATION
    def num_actions( self ): return self.env.action_space.shape[0]
    def range_actions( self ): return abs(self.env.action_space.high[0])

    ### MAP KEYS
    def map_keys( self , actn ):
        actn = np.clip( actn, self.env.action_space.low[0], self.env.action_space.high[0])
        return np.expand_dims(actn,0)

    ### PROCESS OBSERVATION
    def process( self , obsv ):

        return obsv
