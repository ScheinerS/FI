#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt


import aux

def save_results(data:dict, filename:str='data'):
    '''
    filename='data'
    '''
    filename = filename + '.xlsx'
    print('\nOutput file:\t', filename)
    
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    
    for noise in data.keys():
        data[noise].to_excel(writer, sheet_name=noise, header=True)#, index=False)
    
    writer.save()




def save_matrix(m, d:int, filename:str='m', path:str='m', noise:str='NOISE'):
    '''
    filename='m'
    '''
    filename = filename + '.csv'
    aux.check_dir(path)
    plt.savefig(path + os.sep + 'epsilon(eta)_d=%d_%s.pdf'%(d, noise))