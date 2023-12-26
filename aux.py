#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:24:23 2023

@author: santiago
"""
import os
import itertools
import pandas as pd
from datetime import datetime
import shutil
import sys



def check_dir(path):
    directories = path.split(os.sep)
    [directories.insert(2*i+1, os.sep) for i in range(len(directories)-1)]
    directories = list(itertools.accumulate(directories))
    
    for d in directories:
        is_dir = os.path.isdir(d)
        if(not is_dir):
            os.mkdir(d)
            print('Created:', d)


def make_path(L:list):
    '''
    Parameters
    ----------
    L : list
        List of strings to join as path.

    Returns
    -------
    String to be used as path.
    '''
    
    return os.sep.join(L) + os.sep

def get_date():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def save_parameters(directory):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    shutil.copy2('parameters.csv', directory + os.sep + 'parameters_' + timestamp + '.csv')




def read_parameters(filename):
    filename = 'parameters.txt'
    
    P = pd.read_csv(filename, delimiter='=', skiprows=0)#.transpose()
    P['value'] = pd.to_numeric(P['value'], downcast='integer', errors='ignore')
    
    parameters = dict(zip(P['parameter'], P['value']))
    return parameters




def verify_parameters(parameters):
    print(50*'*')
    for item in parameters:
         print(item, '=', parameters[item])
    print(50*'*')

    X = input('Continue (y/n)?')
    if X == 'y':
        return
    elif X == 'n':
        sys.exit()
    else:
        verify_parameters(parameters)






if __name__ == '__main__':
    
    if 1:
        L = ['a', 'b', 'c']
        make_path(L)
    
    if 0:
        path = 'a/b/c/d/e'
        check_dir(path)
    
    if 0:
        filename = 'parameters.txt'
        parameters = read_parameters(filename)
        verify_parameters(parameters)
