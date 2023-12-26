#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 11:29:45 2023

@author: santiago
"""
import os
import aux
import glob
import kraus_operators as ko
import plot_functions as pf
# import animation
import imageio.v3 as iio

plot = 0
animate = 1
D = range(1, 2)

if __name__=='__main__':
    
    eta = 0.5    
    
    for d in D:
        path_read = aux.make_path(['plots', 'Kraus_Operators', 'd=%d'%d])
        path_save = aux.make_path(['animations', 'Kraus_Operators'])
        if plot:        
            # print('\rd=%d\ti=%3d'%(d, i), end='')
            K = ko.amplitude_damping_Kraus_operators(d, eta)
            
            aux.check_dir(path_save)
            for i in range(len(K)):
                print('\rd=%d\ti=%3d'%(d, i), end='')
                pf.plot_matrix(K[i],
                               title = 'K[%d]'%i,
                               save = True,
                               save_name = 'K[%06d]'%i,
                               path = path_save,
                               colour = 'YlGnBu',
                               clim = [-0.05, 1.05],
                               save_as = 'png')
    
        if animate:
            
            files = glob.glob(path_read + '*.png')
            files = sorted(files)
            
            FPS = 2
            initial_seconds = 2
            final_seconds = 2
            
            print('d=%d\tfps=%d'%(d, FPS))
            
            images = []
            for i in range(FPS*initial_seconds):
                images.append(iio.imread(files[0]))
                    
            for file in files:
                images.append(iio.imread(file))
        
            for i in range(FPS*final_seconds):
                images.append(iio.imread(file))
        
            kargs = { 'fps': FPS,
                     # 'quality': 10,
                     'macro_block_size': None,
                     # 'ffmpeg_params': ['-s','600x450']
                     }
            # imageio.mimsave(gifOutputPath, images, 'FFMPEG', **kargs)
            iio.imwrite(path_save + 'KrausOperators_d=%d.mp4'%(d), images, **kargs)

            # animation.animate(state = 'Amplitude Damping Kraus operators',
                              # d = d,
                              # var = 'KO',
                              # noise_types = ['Amplitude Damping'],
                              # FPS = 4,
                              # )
