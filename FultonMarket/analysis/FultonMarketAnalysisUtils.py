# Imports
import os, sys, math, glob
from datetime import datetime
import netCDF4 as nc
import numpy as np
from pymbar import timeseries, MBAR
import scipy.constants as cons
import mdtraj as md
#import dask.array as da
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from sklearn.decomposition import PCA
from pymbar.timeseries import detect_equilibration

fprint = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))



@staticmethod
def resample_with_MBAR(objs: List, u_kln: np.array, N_k: np.array, size: int, reshape_weights: tuple=None, specify_state: int=0, return_inds: bool=False, return_weights: bool=False, return_resampled_weights: bool=False, replace: bool=True):

    # Get MBAR weights
    weights = compute_MBAR_weights(u_kln, N_k)

    # Reshape weights if specified
    if reshape_weights is not None:
        weights = weights.reshape(reshape_weights)
        
    # Get probabilities
    if len(weights.shape) == 1:
        probs = weights.copy()
    else:
        probs = weights[:, specify_state]


    # Resample
    if size == -1:
        size = len(np.where(probs >= probs.max()*0.001)[0])
        fprint(f'Top 99.9% of probability includes {size} no. of frames')
        probs[probs < probs.max()*0.001] = 0
        probs /= probs.sum()
    resampled_inds = np.random.choice(range(len(probs)), size=size, replace=replace, p=probs)
    resampled_objs = []
    for obj in objs:
        resampled_objs.append(np.array([obj[resampled_ind] for resampled_ind in resampled_inds]))

    # Return resampled objects
    return_list = []
    if len(objs) == 1:
        return_list.append(resampled_objs[0])
    elif len(objs) > 1:
        for resampled_obj in resampled_objs:
            return_list.append(resampled_obj)

    # Optional returns
    if return_inds:
        return_list.append(resampled_inds)
    if return_weights:
        return_list.append(weights)
    if return_resampled_weights:
        resampled_weights = weights[resampled_inds, specify_state]
        return_list.append(resampled_weights)

    return return_list


@staticmethod
def compute_MBAR_weights(u_kln, N_k):
    """
    """
    mbar = MBAR(u_kln, N_k, initialize='BAR')

    return mbar.weights()


@staticmethod
# PCA
def detect_PC_equil(pc, reduced_cartesian):
    t0, _, _ = detect_equilibration(reduced_cartesian[:,pc])

    return t0*10

def get_energies_without_harmonic(energies, pos, centers, T, spring_constant):
    x_dis = np.sum((centers[:,0] - pos[:,0])**2, axis=0)
    y_dis = np.sum((centers[:,1] - pos[:,1])**2, axis=0)
    z_dis = np.sum((centers[:,2] - pos[:,2])**2, axis=0)
    displacement = np.sum((x_dis, y_dis, z_dis), axis=0)
    corrected_energies = energies - (1 / (2 * 8.3145 * T)) * spring_constant * displacement

    return corrected_energies