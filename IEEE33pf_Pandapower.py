# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:24:04 2023

@author: diego
"""

import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


feeder = pd.read_csv("IEEE33Datacsv.csv")
num_lines = len(feeder)

net = pp.create_empty_network()

num_nodes=0
for k in range(num_lines):
    num_nodes=max(num_nodes,feeder['i'][k],feeder['j'][k])

Vbase=12.66
Sbase=1e6    
b={}

for k in range(num_nodes): 
    b[k] = pp.create_bus(net, vn_kv=Vbase)
    
for k in range(num_lines):
   pp.create_line_from_parameters(net, from_bus=b[feeder['i'][k]-1], to_bus=b[feeder['j'][k]-1], length_km=1,r_ohm_per_km=feeder['r'][k],x_ohm_per_km=feeder['x'][k],c_nf_per_km=0, max_i_ka=1000)   
   pp.create_load(net, bus=b[feeder['j'][k]-1], p_mw=(feeder['pj'][k])*1e3/Sbase,q_mvar=(feeder['qj'][k])*1e3/Sbase)

pp.create_ext_grid(net, bus=b[0],vm_pu=1.0)


pp.runpp(net)

v=net.res_bus.vm_pu
plt.plot(v)

pg=net.res_ext_grid.p_mw
pg=pg.to_numpy()
pg=pg[0]
qg=net.res_ext_grid.q_mvar
qg=qg.to_numpy()
qg=qg[0]

pl=sum(net.res_line.p_to_mw)+sum(net.res_line.p_from_mw) 