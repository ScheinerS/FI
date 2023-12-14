
import numpy as np
import os

#%% Parameters

d = 3   # Number of qubits.

state = 'GHZplus' # states = ['GHZ', 'plus', 'GHZplus']

n = 1 # number of states to simulate for each combination of eta, etc.
n_rand_states = 1 # number of random states per value of eta. TODO: adapt for this.

n_eta = 1
eta_values = np.linspace(0, 1, num=n_eta)

n_alpha = 200
alpha_values = np.linspace(0, 1, num=n_alpha)

plots_path = 'plots' + os.sep + state + os.sep + 'd=' + str(d) + os.sep  # all plots will be saved in this directory.

colours = {'H': 'coolwarm',
           'M': 'coolwarm',
           'rho_re': 'inferno',
           'rho_im': 'cividis',
           'C': 'viridis'}

colourbar_limits = {'H': [-1/2, 1/2],
                    'M': [-1, 1],
                    'rho_re': [-1, 1],
                    'rho_im': [-1, 1],
                    'C': [-1, 1]}
#%% Flags

flags = {'print_states': 0,
         
         'plot_H_and_M': 0,
         'save_H_and_M': 0,
         
         'plot_rho': 0,
         'save_rho': 1,
         'save_rho_csv': 1,
         
         'plot_C': 0,
         'save_C': 0,
         
         'plot_epsilon': 1,
         'save_epsilon': 1,
         }