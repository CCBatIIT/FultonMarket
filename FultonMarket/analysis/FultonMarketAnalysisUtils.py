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

printf = lambda my_string: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + ' // ' + str(my_string), flush=True)
get_kT = lambda temp: temp*cons.gas_constant
geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]
rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))


def PCA_convergence_detection(rc, rc_err):

    converged = np.array([False for i in range(len(rc)-1)])
    for i, (rc_i, rc_err_i) in enumerate(zip(rc[:-1], rc_err[:-1])):
        if rc_i - rc_err_i <= 0:
            converged[i] = True
        else:
            converged[:i+1] = False

    return converged



def write_traj_from_pos_boxvecs(pos, box_vec, pdb_in, dcd_out):
    # Create traj obj
    traj = md.load_pdb(pdb_in)
    
    # Apply pos, box_vec to mdtraj obj
    traj.xyz = pos.copy()
    traj.unitcell_vectors = box_vec.copy()
    traj.save_dcd(dcd_out)
    
    # Correct periodic issues
    traj = md.load(dcd_out, top=pdb_in)
    prot_sele = traj.topology.select('protein')
    traj = traj.superpose(traj, atom_indices=prot_sele, ref_atom_indices=prot_sele)
    traj.image_molecules()

    return traj



def get_traj_PCA(traj, explained_variance_threshold: float=None):
    """
    """
    # PCA
    pca = PCA()
    reduced_cartesian = pca.fit_transform(traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3))
    explained_variance = np.array([np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(pca.n_components_)])

    if explained_variance_threshold is not None:
        n_components = int(np.where(explained_variance >= explained_variance_threshold)[0][0])

    return pca, reduced_cartesian[:,:n_components], explained_variance[:n_components], n_components    


def calculate_weighted_rc(reduced_cartesian, resampled_inds, upper_limit, pca_weights, mbar_weights):
    assert reduced_cartesian.shape[0] == len(resampled_inds), f'{reduced_cartesian.shape}, {resampled_inds.shape}'
    assert reduced_cartesian.shape[0] == len(mbar_weights)
    assert reduced_cartesian.shape[1] == len(pca_weights)

    mean_weighted_rcs = []
    mean_weighted_rcs_err = []
    for (rc, frame_no, mbar_weight) in zip(reduced_cartesian, resampled_inds[:,1], mbar_weights):
        if frame_no <= upper_limit:
            mean_weighted_rcs.append(np.mean(np.dot(rc, pca_weights)) * mbar_weight)
            mean_weighted_rcs_err.append(np.std(np.dot(rc, pca_weights)) * mbar_weight)
            
    return np.sum(mean_weighted_rcs), np.sqrt(np.sum(np.array(mean_weighted_rcs_err)**2))
    

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
        resampled_inds = np.where(probs >= probs.max()*0.001)[0]
        printf(f'Top 99.9% of probability includes {len(resampled_inds)} no. of frames')
    else:
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

    return t0

def get_energies_without_harmonic(energies, pos, centers, T, spring_constant):
    x_dis = np.sum((centers[:,0] - pos[:,0])**2, axis=0)
    y_dis = np.sum((centers[:,1] - pos[:,1])**2, axis=0)
    z_dis = np.sum((centers[:,2] - pos[:,2])**2, axis=0)
    displacement = np.sum((x_dis, y_dis, z_dis), axis=0)
    corrected_energies = energies - (1 / (2 * 8.3145 * T)) * spring_constant * displacement

    return corrected_energies