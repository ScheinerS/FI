#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:58:25 2023

@author: santiago
"""



import numpy as np
import matplotlib.pyplot as plt


def plot_matrix_3d(m):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(range(0, len(m)), float)
    y = x.copy()
    xpos, ypos = np.meshgrid(x, y)
    z = np.array(m).reshape(-1)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = 0.75 * np.ones_like(zpos)
    dy = dx.copy()
    dz = z.flatten()
    
    ## Define a colorbar
    cmap = plt.cm.get_cmap('viridis')
    max_height = m.max()
    min_height = m.min()
    color_values = [cmap((i-min_height)/max_height) for i in dz]
    
    ## 3D Bar Plot
    bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=color_values)
    
    ## Colorbar
    cbar_obj = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_height, max_height))
    cbar_obj.set_array([dz])
    cbar = plt.colorbar(cbar_obj)
    
    plt.show()


if __name__=='__main__':
    
    plt.close('all')
    M = np.load('density_matrix.npy')
    
    R = np.real(M)
    I = np.imag(M)
    
    plot_matrix_3d(R)
    plot_matrix_3d(I)