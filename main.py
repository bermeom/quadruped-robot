#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
import numpy as np
import mujoco_py as mpy #import load_model_from_path, MjSim, MjViewer
from pprint import pprint
import os

model = mpy.mujoco_py.load_model_from_path("xml/quadrepedrobot.xml")
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
while True:
    sim.set_state(sim_state)
    print("qfrc_applied[0]-> ",sim.data.qfrc_applied);
    force[1]=150;
    # # mpy.functions.mj_applyFT(sim.model,sim.data,force,torque,point,body,sim.data.qfrc_applied);
    # sim.step();
    # print("qfrc_applied[1]-> ",sim.data.qfrc_applied);
    # # mpy.functions.mj_applyFT(sim.model,sim.data,-sim.data.qfrc_applied,torque,point,body,sim.data.qfrc_applied);
    # force[1]=0;
    # sim.step();
    # sim.data.qfrc_applied[0]=0;
    # mpy.functions.mjv_applyPerturbForce(sim.model,sim.data,perp);
    # print("qfrc_applied[2]-> ",sim.data.qfrc_applied);

    for i in range(1000):
        if i < 200 | i > 800 :
            sim.data.ctrl[k] = -0.5
            sim.data.ctrl[k+4] = 0.5
            sim.data.ctrl[k+2*4] = -0.5
            sim.data.ctrl[k+3*4] =  0.5
        else:
            sim.data.ctrl[k] = 1.0/2
            sim.data.ctrl[k+4] = -1.0/2
            sim.data.ctrl[k+2*4] = 1.0/2
            sim.data.ctrl[k+3*4] = -1.0/2
        sim.step();
        viewer.render();
        # mpy.functions.mj_applyFT(sim.model,sim.data,force0,torque,point,body,qfrc_target);
        # print("qfrc_target",qfrc_target);


    if os.getenv('TESTING') is not None:
        break