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

class QuadrupedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    vel_x_ref = 1;
    vel_y_ref = 0;
    err_vel_x = 0;
    err_vel_y = 0;
    
    def __init__(self):
        xmlpath = ((os.path.join((os.path.split(os.path.abspath(__file__))[0]), "xml/quadrupedrobot.xml")));
        mujoco_env.MujocoEnv.__init__(self, xmlpath, 5)
        utils.EzPickle.__init__(self)
        self.vel_x_ref = 1;
        self.vel_y_ref = 0;
        self.err_vel_x = 0;
        self.err_vel_y = 0;
        
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
        orientation = qpos_after.flat[3:7]; # w x y z
        
        # print("qpos : ",qpos_before[0:4]," ",qpos_after[0:4])
        # print("xpos : ",xpos_before," ",xpos_after)
        # print("qvel : ",qvel_before[0:3]," ",qvel_after[0:3])
        # print("mc : ",mc_after," ",mc_after)
        
        alive_bonus = 2.0
        data = self.sim.data
        self.vel_x =  (qpos_after[0] - qpos_before[0])/self.dt;
        self.vel_y =  (qpos_after[1] - qpos_before[1])/self.dt;
        # print("veel",[self.vel_x,self.vel_y ]," vs ",qvel_after[0:2])
        self.err_vel_x = (self.vel_x_ref-self.vel_x);
        self.err_vel_y = (self.vel_y_ref-self.vel_y);
        lin_vel_cost =  self.err_vel_x*self.err_vel_x + self.err_vel_y*self.err_vel_y;
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum();
        quad_impact_cost = .5e-7 * np.square(data.cfrc_ext).sum();
        quad_impact_cost = min(quad_impact_cost, 10.0);
        quad_impact_cost = quad_impact_cost*quad_impact_cost;
        orie_cost = math.fabs(orientation[1]);
        reward = alive_bonus-((1.5*lin_vel_cost + quad_ctrl_cost + quad_impact_cost+orie_cost));
        done = bool((math.fabs(orientation[1])+math.fabs(orientation[2]))>0.6) # rotational angles |x|+|y| 
        # print("Done : ",done," ",reward,lin_vel_cost,quad_ctrl_cost,quad_impact_cost,"vel [",self.err_vel_x,",",self.err_vel_y,"]");
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
                                self.sim.data.cfrc_ext.flat,
                                [self.err_vel_x,self.err_vel_y]
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
