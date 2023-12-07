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

def plot_matrix(m, plot:str='3d', title:str='', save:bool=0, save_name:str='m', path:str='m', colour:str='binary', clim:list=[None, None]):
    
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
        min_height = m.min() # min(m.min(), clim[0])
        max_height = m.max() # max(m.max(), clim[1])
        colour_values = [cmap((i-min_height)/max_height) for i in dz]
        
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


'''
m = np.matrix('0.5 0 0 0.5; 0 0 0 0; 0 0 0 0; 0.5 0 0 0.5')
plot_matrix_3d(m, save=0, colour='binary', clim=[-1, 1])


'''

#%%

import pandas as pd





# ax.plot_surface(x,y,z, cmap=plt.cm.coolwarm,
                       # linewidth=0, antialiased=False)
#%%
# surf = ax.plot_trisurf(x, y, z, cmap= plt.cm.coolwarm, linewidth=0.2) 







# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')



# #ax.plot_wireframe(cParams, gammas, avg_errors_array)
# #ax.plot3D(cParams, gammas, avg_errors_array)
# #ax.scatter3D(cParams, gammas, avg_errors_array, zdir='z',cmap='viridis')

# # df = pd.DataFrame({'x': cParams, 'y': gammas, 'z': avg_errors_array})
# surf = ax.plot_trisurf(x, y, z, cmap=plt.cm.jet, linewidth=0.1)
# fig.colorbar(surf, shrink=0.5, aspect=5)    
# # plt.savefig('./plots/avgErrs_vs_C_andgamma_type_%s.png'%(k))
# plt.show()






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
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array(data[noise]['alpha'], dtype='float64')
        y = np.array(data[noise]['eta'], dtype='float64')
        z = np.array(data[noise]['epsilon'], dtype='float64')
        
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

#%%


import numpy as np
import numpy.random
import matplotlib.pyplot as plt

# To generate some test data
x = np.random.randn(500)
y = np.random.randn(500)

XY = np.stack((x,y),axis=-1)

def selection(XY, limitXY=[[-2,+2],[-2,+2]]):
        XY_select = []
        for elt in XY:
            if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
                XY_select.append(elt)

        return np.array(XY_select)

XY_select = selection(XY, limitXY=[[-2,+2],[-2,+2]])


xAmplitudes = np.array(XY_select)[:,0]#your data here
yAmplitudes = np.array(XY_select)[:,1]#your other data here





def plot_matrix_3d(m):
    
    d = 4
    m = np.zeros((2**d,2**d))
    m[0][0] = 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.bar
    # hist, xedges, yedges = np.histogram2d(x, y, bins=(7,7), range = [[-2,+2],[-2,+2]]) # you can change your bins, and the range on which to take data
    # # hist is a 7X7 matrix, with the populations for each of the subspace parts.
    # xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) -(xedges[1]-xedges[0])
    
    
    # xpos = xpos.flatten()*1./2
    # ypos = ypos.flatten()*1./2
    # zpos = np.zeros_like (xpos)
    
    # dx = xedges [1] - xedges [0]
    # dy = yedges [1] - yedges [0]
    # dz = hist.flatten()
    
    # cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
    # max_height = np.max(dz)   # get range of colorbars so we can normalize
    # min_height = np.min(dz)
    # # scale each z to [0,1], and get their rgb values
    # rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    # # plt.title("X vs. Y Amplitudes for ____ Data")
    # plt.xlabel("X")
    # plt.ylabel("Y ")
    # plt.savefig("TITLE.pdf")
    # plt.show()
    

#%%



if __name__=='__main__':
    
    plt.close('all')
    M = np.load('density_matrix.npy')
    
    R = np.real(M)
    I = np.imag(M)
    
    plot_matrix_3d(R)
    plot_matrix_3d(I)
    