import numpy as np
from gym import utils
from os import path
from gym.envs.mujoco import mujoco_env
import math
import os

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class QuadrepedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xmlpath = ((os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "xml/quadrepedrobot.xml")));
        mujoco_env.MujocoEnv.__init__(self, xmlpath, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        orientation = qpos.flat[3:7]; # w x y z
        done = bool((math.fabs(orientation[1])+math.fabs(orientation[2]))>0.5) # rotational angles |x|+|y| 
        print("Done : ",done," ",orientation)
        # done = False;
        ob = self._get_obs()
        return ob, reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)
        # print("reward : ",dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost));
        # xposbefore = self.sim.data.qpos[0]
        # self.do_simulation(action, self.frame_skip)
        # xposafter = self.sim.data.qpos[0]
        # ob = self._get_obs()
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        # reward = reward_ctrl + reward_run
        # done = False
        # return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
                               self.sim.data.qpos.flat[1:],
                               self.sim.data.qvel.flat,
                               self.sim.data.cinert.flat,
                               self.sim.data.cvel.flat,
                               self.sim.data.qfrc_actuator.flat,
                               self.sim.data.cfrc_ext.flat
        ])

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
