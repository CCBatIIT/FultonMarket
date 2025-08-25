import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('pdb_dir', help="path to directory with .pdb files for each simulation in 'repexchange_dir'")
parser.add_argument('repexchange_dir', help="path to directory with the replica exchange output directories")
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
def perform_truncation(dir, pdb):
   
    # Initialize
    analysis = FultonMarketAnalysis(dir, pdb)

    # Truncate
    analysis.truncate()
    
    del analysis    


if __name__ == '__main__':
    
    # Input
    sims = sorted(os.listdir(args.repexchange_dir))
    repexchange_dir = args.repexchange_dir
    pdb_dir = args.pdb_dir

    # Set up arguments
    mpargs = []
    for sim in sims:
        
        # Sim input
        dir = os.path.join(repexchange_dir, sim)
        pdb = os.path.join(pdb_dir, "_".join(sim.split('_')[:-1]) + '.pdb')
        perform_truncation(dir, pdb)


