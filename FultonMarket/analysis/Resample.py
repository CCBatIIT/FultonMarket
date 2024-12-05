from analysis.FultonMarketAnalysis import FultonMarketAnalysis
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymbar.timeseries import detect_equilibration
import os
import netCDF4 as nc

# Input
sims = [f'centroid_{i}_0' for i in range(1,21)]
repexchange_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/PTwFR'
pdb_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/centroids'
output_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/resampled/PTwFR'

for sim in sims:
    # Define outputs
    pdb_out = os.path.join(output_dir, 'pdb', "_".join(sim.split('_')[:-1]) + '.pdb')
    dcd_out = os.path.join(output_dir, 'dcd', "_".join(sim.split('_')[:-1]) + '.dcd')
    
    # Sim input
    print(sim)
    dir = os.path.join(repexchange_dir, sim)
    pdb = os.path.join(pdb_dir, sim.split('_')[0] + '.pdb')
    
    # Make obj
    analysis = FultonMarketAnalysis(dir, pdb, skip=10, remove_harmonic=True, resids=np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311]))
    
    # Importance Resampling
    analysis.importance_resampling(equilibration_method='energy')
    analysis.write_resampled_traj(pdb_out, dcd_out)