#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:24:23 2023

@author: santiago
"""

import os

def check_dir(path):
    is_dir = os.path.isdir(path)
    if(not is_dir):
        os.mkdir(path)
        print('Created:', path)