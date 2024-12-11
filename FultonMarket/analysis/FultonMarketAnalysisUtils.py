# Imports
import os, sys, math, glob, jax, itertools
import jax.numpy as jnp
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
perms = jnp.array([x for x in itertools.product([-1, 0, 1], repeat=3)])
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))
jaxrmsd = lambda a, b: jnp.sqrt(jnp.mean(jnp.sum((b-a)**2, axis=-1), axis=-1))
fprint = lambda x: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + x, flush=True)
jax_add = lambda a, b: a+b
jax_add = jax.vmap(jax_add, in_axes=(0, None))
rmsd_j = jax.vmap(jaxrmsd, in_axes=(0, None))



@staticmethod
def resample_with_MBAR(objs: List, u_kln: np.array, N_k: np.array, size: int, reshape_weights: tuple=None, specify_state: int=0, return_inds: bool=False, return_weights: bool=False):

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
    resampled_inds = np.random.choice(range(len(probs)), size=size, replace=True, p=probs)
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

    if return_inds:
        return_list.append(resampled_inds)
    if return_weights:
        return_list.append(weights)

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


def get_restraint_energy_kT(pos, trans, centers, T, spring_constant):
    pos += trans # Translate, if necessary, to avoid wrapping issues
    pos *= 10 #Convert to Angstrom
    centers *= 10 #Convert to Angstrom
    x_dis = np.sum((centers[:,0] - pos[:,0])**2, axis=0)
    y_dis = np.sum((centers[:,1] - pos[:,1])**2, axis=0)
    z_dis = np.sum((centers[:,2] - pos[:,2])**2, axis=0)
    displacement_sq = np.sum((x_dis, y_dis, z_dis), axis=0)

    restraint_energy = (1 / (2 * 8.3145 * T)) * spring_constant * displacement_sq # E(kT) = (1/2RT) * k_spring * ((x-x0)**2 + (y-y0)**2 + (z-z0)**2)  This expression converts restraint energies in (J/mol) to kT to match openmmtools energies

    return restraint_energy


def best_translation_by_unitcell(cell_lengths, mobile_coords, target_coords):
    translations = cell_lengths * perms
    permuted_positions = jax_add(translations, mobile_coords)
    rmsds_of_permutations = rmsd_j(permuted_positions, target_coords)
    return translations[jnp.argmin(rmsds_of_permutations)], rmsds_of_permutations[jnp.argmin(rmsds_of_permutations)]

# Jax to speed up some functions
best_translation_by_unitcell = jax.vmap(best_translation_by_unitcell, in_axes=(0, 0, None))