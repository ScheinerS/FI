#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import aux

def save_results(data:dict, filename:str='data', path=''):
    
    filename = filename + '.xlsx'
    print('\nOutput file:\t', filename)
    
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    
    for noise in data.keys():
        data[noise].to_excel(writer, sheet_name=noise, header=True)#, index=False)
    
    writer.close()




def save_matrix(m, d:int, eta:float=0, alpha:float=0, filename:str='m', path:str='m', noise:str='NOISE'):
    '''
    filename='m'
    '''
    re = np.real(m)
    im = np.imag(m)
    filename = filename + '.csv'
    aux.check_dir(path)
    # Real part:
    np.savetxt(path + os.sep +'rho_eta=%.2f_alpha=%.2f_re'%(eta, alpha) + '.txt',
               re,
               delimiter="\t",
               fmt='%.4f'
               )
    # Imaginary part:
    np.savetxt(path + os.sep +'rho_eta=%.2f_alpha=%.2f_im'%(eta, alpha) + '.txt',
               im,
               delimiter="\t",
               fmt='%.4f'
               ) 



if __name__ == '__main__':
    m = [[1, 2], [3, 4]]
    save_matrix(m, 2)