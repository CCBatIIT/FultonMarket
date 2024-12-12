from FultonMarketAnalysis import FultonMarketAnalysis
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymbar.timeseries import detect_equilibration
import os, sys
import netCDF4 as nc
import argparse
import multiprocessing as mp




# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('pdb_dir', help="path to directory with .pdb files for each simulation in 'repexchange_dir'")
parser.add_argument('repexchange_dir', help="path to directory with the replica exchange output directories")
parser.add_argument('output_dir', help="path to outputdir where resampled trajectories will be stored.")
parser.add_argument('resids_npy', help="path to .npy file with the resids to include for principal component analysis and equilibration detection")
parser.add_argument('--upper-limit', default=None, type=int, help="upper limit (number of frames) for resampling. Default is None, meaning all of the frames from replica exchange will be included in resampling.")
parser.add_argument('--parallel', action='store_true', help="choose to multiprocess the calculation across different replica exchange simulations")
args = parser.parse_args()


# Multiprocessing method
def resample(dir, pdb, upper_limit, resids, pdb_out, dcd_out):

    # Initialize
    analysis = FultonMarketAnalysis(dir, pdb, skip=10, upper_limit=upper_limit, resids=resids)
    
    # Importance Resampling
    try:
        analysis.importance_resampling(equilibration_method='PCA')
    except NameError:
        pass    

    # Write out
    analysis.write_resampled_traj(pdb_out, dcd_out)


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

    # Set up arguments
    mpargs = []
    for sim in sims:
        
        # Define outputs
        pdb_out = os.path.join(output_dir, 'pdb', "_".join(sim.split('_')[:-1]) + '.pdb')
        dcd_out = os.path.join(output_dir, 'dcd', "_".join(sim.split('_')[:-1]) + '.dcd')
        if not os.path.exists(dcd_out):
            
            # Sim input
            dir = os.path.join(repexchange_dir, sim)
            pdb = os.path.join(pdb_dir, sim.split('_')[0] + '.pdb')
            upper_limit = args.upper_limit
            resids = np.load(args.resids_npy)

            if parallel:
                mpargs.append((dir, pdb, upper_limit, resids, pdb_out, dcd_out))

            else:
                resample(dir, pdb, upper_limit, resids, pdb_out, dcd_out)

    
    # Multiprocess, if specified
    if parallel:
        with mp.Pool(len(mpargs)) as p:
            p.starmap(resample, mpargs)

