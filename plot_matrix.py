#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:58:25 2023

@author: santiago
"""



import numpy as np
import matplotlib.pyplot as plt
import os

import aux


def plot_matrix_2d(M, title:str='', save:bool=0, save_name:str='M', path:str='M', colour:str='binary', clim:list=[None, None]):
    
    fig, ax = plt.subplots()
    import matplotlib
    im = ax.imshow(M, cmap=colour, norm=matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1]))
    # ax.set_title('Pan on the colorbar to shift the color mapping\n'             'Zoom on the colorbar to scale the color mapping')
    colourbar = fig.colorbar(im, ax=ax, label='')
    # cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cm, norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
    colourbar.ax.set_ylim(clim)
    plt.show()
    
    # plt.imshow(rho,  cmap='Purples')
    # plt.colorbar()
    plt.title(r'%s'%title)
    # plt.show()
    
    '''
    Graficar en 3D.
    '''
    
    if save:
        # aux.check_dir(str(n_qubits)+'_qubits')
        aux.check_dir(path)
        plt.savefig(path + os.sep + save_name + '.pdf')
        print('Saved:\t', path + os.sep + save_name + '.pdf')





        
def plot_matrix_3d(M, title:str='', save:bool=0, save_name:str='M', path:str='M', colour:str='binary', clim:list=[None, None]):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(range(0, len(M)), float)
    y = x.copy()
    xpos, ypos = np.meshgrid(x, y)
    z = np.array(M).reshape(-1)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = 0.75 * np.ones_like(zpos)
    dy = dx.copy()
    dz = z.flatten()
     
    ## Define a colorbar
    cmap = plt.cm.get_cmap(colour)
    [min_height, max_height] = clim
    color_values = [cmap((i-min_height)/max_height) for i in dz]
    
    # Limits for z axis:
    ax.axes.set_zlim3d(bottom=clim[0], top=clim[1])

    ## 3D Bar Plot
    bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=color_values)
    
    ## Colorbar
    cbar_obj = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_height, max_height))
    cbar_obj.set_array([dz])
    cbar = plt.colorbar(cbar_obj)
    
    plt.show()
    
    if save:
        # aux.check_dir(str(n_qubits)+'_qubits')
        aux.check_dir(path)
        plt.savefig(path + os.sep + save_name + '.pdf')
        print('Saved:\t', path + os.sep + save_name + '.pdf')

if __name__=='__main__':
    
    plt.close('all')
    M = np.load('density_matrix.npy')
    
    R = np.real(M)
    I = np.imag(M)
    
    plot_matrix_3d(R)
    plot_matrix_3d(I)