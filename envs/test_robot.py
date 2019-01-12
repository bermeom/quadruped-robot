#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
import numpy as np
import math
from pprint import pprint
import mujoco_py as mpy #import load_model_from_path, MjSim, MjViewer
import os

   
model = mpy.mujoco_py.load_model_from_path("xml/quadrupedrobot.xml")
# model = mpy.mujoco_py.load_model_from_path("xml/humanoid.xml")
sim = mpy.MjSim(model)

viewer = mpy.MjViewer(sim)


sim_state = sim.get_state()

pprint (sim_state);

k = 1;
qfrc_target = sim.data.qfrc_applied;
force = qfrc_target;#np.array([0, +150, 0],dtype=np.float64);
# force[1]=150;
force0 = np.array([0, -150, 0],dtype=np.float64);
torque = np.array([0, 0, 0],dtype=np.float64);
point = np.array([0, 0, 0],dtype=np.float64) ;
body = 1;
qfrc_target = sim.data.qfrc_applied;
perp = mpy.cymj.PyMjvPerturb();
# print("start",qfrc_target);
sw = True;
while True:
    print ("0 - orientation -> body_quat ",sim.data.qpos.flat[3:7])
    # print ("0 - orientation -> body_quat ",sim.data.qpos.flat[0:7])
    mat = np.zeros(9, dtype=np.float64)
    mpy.functions.mju_quat2Mat(mat, sim.data.body_xquat[2]);
    # print ("0 - orientation -> ",mat)
    # print("0 - data -> ",sim.data.qM)
    
    # sw = False;
    sim.set_state(sim_state)
    print ("0 - orientation -> body_quat ",sim.data.qpos.flat[3:7])
    mat = np.zeros(9, dtype=np.float64)
    mpy.functions.mju_quat2Mat(mat, sim.data.body_xquat[2]);
    # print ("orientation -> ",mat)
    # print("data -> ",sim.data.qM)
    # print("qfrc_applied[0]-> ",sim.data.qfrc_applied);
    force[1]=-150;
    # mpy.functions.mj_applyFT(sim.model,sim.data,force,torque,point,body,sim.data.qfrc_applied);
    sim.step();
    # print("qfrc_applied[1]-> ",sim.data.qfrc_applied);
    # mpy.functions.mj_applyFT(sim.model,sim.data,-sim.data.qfrc_applied,torque,point,body,sim.data.qfrc_applied);
    if sw :
        force[1]=0;
    sw =  not sw;
    # sim.step();
    # sim.data.qfrc_applied[0]=0;
    # mpy.functions.mjv_applyPerturbForce(sim.model,sim.data,perp);
    # print("qfrc_applied[2]-> ",sim.data.qfrc_applied);
    
    # pprint(sim.data.qpos.flat)   

    # sim.data.ctrl[4*0+3]=-1;
    # sim.data.ctrl[4*1+3]=-1;
    # sim.data.ctrl[4*2+3]=-1;
    # sim.data.ctrl[4*3+3]=-1;
    sim.step();
    observations = np.concatenate([
            sim.data.qpos,
            sim.data.qvel,
        ]);
    orientation = sim.data.qpos.flat[3:7];
    # print(len(sim.data.qpos)," ",len(sim.data.qvel),"\n",observations);
    for i in range(500):
        if i < 200 | i > 800 :
            sim.data.ctrl[3]=1;
            sim.data.ctrl[k] = -0.5
            sim.data.ctrl[k+4] = 0.5
            sim.data.ctrl[k+2*4] = -0.5
            sim.data.ctrl[k+3*4] =  0.5
        else:
        #     sim.data.ctrl[3]=-1;
            sim.data.ctrl[k] = 1.0/2
            sim.data.ctrl[k+4] = -1.0/2
            sim.data.ctrl[k+2*4] = 1.0/2
            sim.data.ctrl[k+3*4] = -1.0/2
        sim.step();
        viewer.render();
        orientation = sim.data.qpos.flat[3:7]; # w x y z
        if (orientation[1]+orientation[2]>0.5):
            break;

        # mpy.functions.mj_applyFT(sim.model,sim.data,force0,torque,point,body,qfrc_target);
        # print("qfrc_target",qfrc_target);
    observations = np.concatenate([
        sim.data.qpos.flat[1:],
        sim.data.qvel.flat,
    ]);
    # print("END",len(sim.data.qpos)," ",len(sim.data.qvel),"\n",observations);
    # mat = np.zeros(9, dtype=np.float64)
    # mpy.functions.mju_quat2Mat(mat, sim.data.body_xquat[2]);
    # print ("orientation -> ",mat)
    # print("data -> ",sim.data.qM)
    
    
    if os.getenv('TESTING') is not None:
        break