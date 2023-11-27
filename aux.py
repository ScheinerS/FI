#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:24:23 2023

@author: santiago
"""
import os
import itertools

def check_dir(path):
    directories = path.split(os.sep)
    [directories.insert(2*i+1, os.sep) for i in range(len(directories)-1)]
    directories = list(itertools.accumulate(directories))
    
    for d in directories:
        is_dir = os.path.isdir(d)
        if(not is_dir):
            os.mkdir(d)
            print('Created:', d)

if __name__ == '__main__':
    path = 'a/b/c/d/e'
    check_dir(path)
