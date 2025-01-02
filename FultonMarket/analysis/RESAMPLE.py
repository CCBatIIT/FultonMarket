import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('pdb_dir', help="path to directory with .pdb files for each simulation in 'repexchange_dir'")
parser.add_argument('repexchange_dir', help="path to directory with the replica exchange output directories")
parser.add_argument('output_dir', help="path to outputdir where resampled trajectories will be stored.")
parser.add_argument('resids_npy', help="path to .npy file with the resids to include for principal component analysis and equilibration detection")
parser.add_argument('--nframes', default=-1, type=int, help="number of frames to resample, default is -1 meaning to resample frames from the top 99.9% of probability.")
parser.add_argument('--no-replace', action='store_true', help="choose not to use resampling with replacement. this is only recommended when n_frames is -1 or default")
parser.add_argument('--upper-limit', default=None, type=int, help="upper limit (number of frames) for resampling. Default is None, meaning all of the frames from replica exchange will be included in resampling.")
parser.add_argument('--parallel', action='store_true', help="choose to multiprocess the calculation across different replica exchange simulations")
args = parser.parse_args()

from FultonMarketAnalysis import FultonMarketAnalysis
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymbar.timeseries import detect_equilibration
import os, sys
import netCDF4 as nc 
import multiprocessing as mp
import jax


# Multiprocessing method
def resample(dir, pdb, upper_limit, resids, pdb_out, dcd_out, weights_out, n_samples, replace):

   
    # Initialize
    analysis = FultonMarketAnalysis(dir, pdb, skip=10, upper_limit=upper_limit, resids=resids)
    
    # Importance Resampling
    try:
        analysis.importance_resampling(n_samples=n_samples, equilibration_method='PCA', replace=replace)
    except NameError:
        pass    

    # Write out
    analysis.write_resampled_traj(pdb_out, dcd_out, weights_out)
    del analysis    


if __name__ == '__main__':
    
    # Input
    sims = sorted(os.listdir(args.repexchange_dir))
    repexchange_dir = args.repexchange_dir
    output_dir = args.output_dir
    pdb_dir = args.pdb_dir
    parallel = args.parallel
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'pdb')):
        os.mkdir(os.path.join(output_dir, 'pdb'))
    if not os.path.exists(os.path.join(output_dir, 'dcd')):
        os.mkdir(os.path.join(output_dir, 'dcd'))
    if not os.path.exists(os.path.join(output_dir, 'mbar_weights')):
        os.mkdir(os.path.join(output_dir, 'mbar_weights'))

    # Set up arguments
    mpargs = []
    for sim in sims:
        
        # Define outputs
        pdb_out = os.path.join(output_dir, 'pdb', "_".join(sim.split('_')[:-1]) + '.pdb')
        dcd_out = os.path.join(output_dir, 'dcd', "_".join(sim.split('_')[:-1]) + '.dcd')
        weights_out = os.path.join(output_dir, 'mbar_weights', "_".join(sim.split('_')[:-1]) + '.npy')
        if not os.path.exists(dcd_out):
            
            # Sim input
            dir = os.path.join(repexchange_dir, sim)
            pdb = os.path.join(pdb_dir, sim.split('_')[0] + '.pdb')
            upper_limit = args.upper_limit
            resids = np.load(args.resids_npy)
            n_samples = args.nframes
            if args.no_replace:
                replace = False
            else:
                replace = True

            if parallel:
                mpargs.append((dir, pdb, upper_limit, resids, pdb_out, dcd_out, weights_out, n_samples, replace))

            else:
                resample(dir, pdb, upper_limit, resids, pdb_out, dcd_out, weights_out, n_samples, replace)

    
    # Multiprocess, if specified
    if parallel:
        counter = 0
        while len(os.listdir(os.path.join(output_dir, 'pdb'))) < len(sims) and counter < 5:
            try:
                n_threads = int(os.environ('NUM_THREADS'))
            except:
                n_threads = 5
            with mp.Pool(n_threads) as p:
                p.starmap(resample, mpargs)
                p.close()
                p.join()

            counter += 1

