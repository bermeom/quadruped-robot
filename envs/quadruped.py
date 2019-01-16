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
    vel_x_ref   = 2;
    vel_y_ref   = 0;
    ori_ref     = 0;
    err_vel_x   = 0;
    err_vel_y   = 0;
    err_vel_x_1 = 0;
    err_ori     = 0;
    err_vel_y_1 = 0;
    err_ori_1   = 0;
    count_step  = 0;

    def __init__(self):
        xmlpath = ((os.path.join((os.path.split(os.path.abspath(__file__))[0]), "xml/quadrupedrobot.xml")));
        mujoco_env.MujocoEnv.__init__(self, xmlpath, 5)
        utils.EzPickle.__init__(self)
        self.vel_x_ref = 1;
        self.vel_y_ref = 0;
        self.err_vel_x = 0;
        self.err_vel_y = 0;
        self.count_step = 0;

    def step(self, action):
        if (len(action)==1):
            action=action[0];

        # for i in range(0,len(action)):
        #     action[i] = np.cos(self.count_step/self.dt+action[i])

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
        rot_axis_z = np.arctan2(2*(orientation[0]*orientation[3]+orientation[1]*orientation[2]),1-2*(orientation[2]*orientation[2]+orientation[3]*orientation[3]));
        # print("qpos : ",qpos_before[0:4]," ",qpos_after[0:4])
        # print("xpos : ",xpos_before," ",xpos_after)
        # print("qvel : ",qvel_before[0:3]," ",qvel_after[0:3])
        # print("mc : ",mc_after," ",mc_after)
        
        data = self.sim.data
        self.vel_x =  (qpos_after[0] - qpos_before[0])/self.dt;
        self.vel_y =  (qpos_after[1] - qpos_before[1])/self.dt;
        # print("veel",[self.vel_x,self.vel_y ]," vs ",qvel_after[0:2])
        self.err_vel_x = (self.vel_x_ref-self.vel_x);
        self.err_vel_y = (self.vel_y_ref-self.vel_y);
        self.err_ori = (self.ori_ref-rot_axis_z);
        
        alive_bonus = 0.0
        bonus_for_doing_well =  (self.err_vel_x_1-self.err_vel_x)+ (self.err_vel_y_1-self.err_vel_y)+ (self.err_ori_1-self.err_ori);
          
        lin_vel_cost =  self.err_vel_x*self.err_vel_x + 0.0*self.err_vel_y*self.err_vel_y;
        quad_ctrl_cost = 0.01 * np.square(data.ctrl).sum();
        quad_impact_cost = .1e-7 * np.square(data.cfrc_ext).sum();
        quad_impact_cost = min(quad_impact_cost, 10.0);
        quad_impact_cost = 0.0*(quad_impact_cost*quad_impact_cost);
        orie_cost =  math.fabs(orientation[1]) + math.fabs(orientation[2]) + math.fabs(self.err_ori);
        # reward = alive_bonus+bonus_for_doing_well-((lin_vel_cost + quad_ctrl_cost + quad_impact_cost+orie_cost));
        c = 0.2;# (1/np.sqrt(2*np.pi*c))*
        vel_x_bonus = np.exp(-math.fabs(self.err_vel_x)/(2*c));
        vel_y_bonus = np.exp(-math.fabs(self.err_vel_y)/(2*c));
        ori_bonus   = np.exp(-math.fabs(self.err_ori)/(2*c));
        ori_cost    = ((np.exp(-(math.fabs(orientation[1]))/(2*c))+np.exp(-(math.fabs(orientation[2]))/(2*c)))/2);
        # reward = alive_bonus+vel_x_bonus+vel_y_bonus+ori_bonus+ori_cost-(quad_ctrl_cost + quad_impact_cost);
        reward = (0.50*vel_x_bonus+0.2*vel_y_bonus+0.15*ori_bonus+0.15*ori_cost)-0.5;
        done = bool((math.fabs(orientation[1])+math.fabs(orientation[2]))>0.5) # rotational angles |x|+|y| 
        # print("Done : ",done," ",reward," ",vel_x_bonus," ",self.vel_x," ",vel_y_bonus," ",ori_bonus," ",ori_cost," ",(quad_ctrl_cost + quad_impact_cost));
        # reward = reward -100*done;
        # print("Done : ",done," ",reward,lin_vel_cost,quad_ctrl_cost,quad_impact_cost,"vel [",self.err_vel_x,",",self.err_vel_y,"]");
        # print("Orientation",RotationAxis_z)
        # done = False;
        ob = self._get_obs()
        self.err_vel_x_1 = self.err_vel_x;
        self.err_vel_y_1 = self.err_vel_y;
        self.err_ori     = self.err_ori;
        self.count_step = ( not done)*(self.count_step+1)
        # print("len",len(ob))
        return ob, reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def _get_obs(self):
        # print("obs -> ",len(self.sim.data.qpos.flat))
        return np.concatenate([
                                self.sim.data.qpos.flat,
                                self.sim.data.qvel.flat,
                                self.sim.data.cinert.flat,
                                self.sim.data.cvel.flat,
                                self.sim.data.qfrc_actuator.flat,
                                self.sim.data.cfrc_ext.flat,
                                [self.err_vel_x,self.err_vel_y,self.err_ori]
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
