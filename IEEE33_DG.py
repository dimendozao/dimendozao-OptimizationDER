# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:24:39 2023

@author: diego
"""

import pandapower as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def IEEE33_DG(net):
    pp.runpp(net,"nr")
    pg=net.res_ext_grid.p_mw
    pg=pg.to_numpy()
    pg=pg[0]
    qg=net.res_ext_grid.q_mvar
    qg=qg.to_numpy()
    qg=qg[0]
    pl=sum(net.res_line.p_to_mw)+sum(net.res_line.p_from_mw)    
    return pg,qg,pl




