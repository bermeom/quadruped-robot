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
        xmlpath = ((os.path.join((os.path.split(os.path.abspath(__file__))[0]), "xml/quadrepedrobot.xml")));
        mujoco_env.MujocoEnv.__init__(self, xmlpath, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        mc_before = mass_center(self.model, self.sim)
        qpos_before = np.array(self.sim.data.qpos);
        qvel_before = np.array(self.sim.data.qvel);
        xpos_before = np.array(self.sim.data.xipos);
        self.do_simulation(action, self.frame_skip)
        mc_after = mass_center(self.model, self.sim)
        qpos_after = np.array(self.sim.data.qpos);
        qvel_after = np.array(self.sim.data.qvel);
        xpos_after = np.array(self.sim.data.xipos); 
        # print("qpos : ",qpos_before[0:4]," ",qpos_after[0:4])
        # print("xpos : ",xpos_before," ",xpos_after)
        # print("qvel : ",qvel_before[0:3]," ",qvel_after[0:3])
        # print("mc : ",mc_after," ",mc_after)
        
        alive_bonus = 1.0
        data = self.sim.data
        vel_x = (qpos_after[0] - qpos_before[0])/self.dt;
        vel_y = (qpos_after[1] - qpos_before[1])/self.dt;
        lin_vel_cost =   (vel_x+vel_y*0)
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = ((lin_vel_cost - quad_ctrl_cost - quad_impact_cost) + alive_bonus)
        orientation = qpos_after.flat[3:7]; # w x y z
        done = bool((math.fabs(orientation[1])+math.fabs(orientation[2]))>0.5) # rotational angles |x|+|y| 
        # print("Done : ",done," ",reward,lin_vel_cost,quad_ctrl_cost,quad_impact_cost,"vel [",vel_x,",",vel_y,"]");
        # done = False;
        ob = self._get_obs()
        return ob, reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

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
