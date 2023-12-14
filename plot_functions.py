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

def plot_matrix(m, plot:str='3d', title:str='', save:bool=0, save_name:str='m', path:str='m', colour:str='binary', clim:list=[0, 0]):
    
    if plot=='2d':
        fig, ax = plt.subplots()
        import matplotlib
        im = ax.imshow(m, cmap=colour, norm=matplotlib.colors.Normalize(vmin=clim[0], vmax=clim[1]))
        # ax.set_title('Pan on the colorbar to shift the color mapping\n'             'Zoom on the colorbar to scale the color mapping')
        colourbar = fig.colorbar(im, ax=ax, label='')
        # cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cm, norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
        colourbar.ax.set_ylim(clim)
        # plt.show()
        
        # plt.imshow(rho,  cmap='Purples')
        # plt.colorbar()
        plt.title(r'%s'%title)
        # plt.show()
    
    elif plot=='3d':
        
        d = int(np.log2(len(m)))
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax = fig.add_subplot(122, projection='3d')
        x = np.array(range(0, 2**d), float)
        y = x.copy()
        xpos, ypos = np.meshgrid(x, y)
        z = np.array(m).reshape(-1)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        dx = 0.75 * np.ones_like(zpos)
        dy = dx.copy()
        dz = z.flatten() # This is the actual data.
        
        cmap = plt.cm.get_cmap(colour)
        min_height = m.min()
        max_height = m.max()
        
        # print(min_height, max_height)
        # min_height = clim[0]
        # max_height = clim[1]
        colour_values = []
        for i in dz:
            if max_height==0:
                colour_values.append(cmap((max_height-min_height)/2))
            else:
                colour_values.append(cmap((i-min_height)/max_height))
        
        cbar_obj = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_height, max_height))
        cbar_obj.set_array([dz])
        cbar = plt.colorbar(cbar_obj)
        
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
                 color=colour_values,
                 shade=True
                 )
        
        plt.show()
    
    if save:
        # aux.check_dir(str(n_qubits)+'_qubits')
        aux.check_dir(path)
        plt.savefig(path + os.sep + save_name + '.pdf')
        # print('\rSaved:\t', path + os.sep + save_name + '.pdf', end='')







def plot_epsilon(d:int, data, noise:str, plot:str='3d', title:str='', save:bool=0, save_name:str='m', path:str='m', state:str='STATE'):
    
    if plot=='2d':
        plt.figure()
        plt.scatter(data[noise]['eta'], data[noise]['epsilon'], c=data[noise]['alpha'], alpha = 1, cmap='Blues')
        plt.colorbar(label=r'$\alpha$')
        plt.title(r'%s'%noise)
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$\epsilon(\eta)$')
        plt.grid(alpha=0.5)
        plt.show()
        if save:
            aux.check_dir(path)
            plt.savefig(path + 'epsilon(eta)_d=%d_%s_%s.pdf'%(d, state, noise))
            
            
        plt.figure()
        plt.scatter(data[noise]['alpha'], data[noise]['epsilon'], c=data[noise]['eta'], alpha = 1, cmap='Purples')
        plt.colorbar(label=r'$\eta$')
        plt.title(r'%s'%noise)
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\epsilon(\alpha)$')
        plt.grid()
        plt.show()
        
    elif plot=='3d':
        
        x = np.array(data[noise]['alpha'], dtype='float64')
        y = np.array(data[noise]['eta'], dtype='float64')
        z = np.array(data[noise]['epsilon'], dtype='float64')
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\eta$')
        ax.set_zlabel(r'$\epsilon$')
        
        plt.title(r'%s'%noise)
        
        ax.plot_trisurf(x, y, z, cmap=plt.cm.jet, linewidth=0.1)
        # ax.scatter(x, y, z)
        # ax.plot_trisurf(x,y,z)
        
    if save:
        aux.check_dir(path)
        plt.savefig(path + 'epsilon(alpha)_d=%d_%s_%s.pdf'%(d, state, noise))



 



if __name__=='__main__':
    
    plt.close('all')
    # M = np.load('density_matrix.npy')

    M = np.array([[0, 1], [1, 0]])
    R = np.real(M)
    I = np.imag(M)
    
    plot_matrix(R, plot='3d', save=0, colour='binary', title='Re', clim=[-0.5, 0.5])
    plot_matrix(I, plot='3d', save=0, colour='binary', title='Im', clim=[-0.5, 0.5])
