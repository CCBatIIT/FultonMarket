from FultonMarketAnalysis import FultonMarketAnalysis
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pymbar.timeseries import detect_equilibration
import os, sys
import netCDF4 as nc

# Input
upper_limit = int(sys.argv[1])
sims = ['7OH_0', '7scg_0', '7u2k_4', '7u2l_6', '8ef5_0', '8ef6_0', '8efb_6', '8efo_0', '8efq_0', '8f7q_0', '8f7r_0', 'LeuEnk_0', 'MetEnk_0', 'apo_0', 'buprenorphine_0', 'c11guano_6', 'c3guano_1', 'c7guano_6', 'c9guano_6', 'carfentanil_0', 'dynorphin_0', 'oxycodone_0', 'pentazocine_0']
repexchange_dir = '/expanse/lustre/projects/iit119/dcooper/MOR/replica_exchange'
pdb_dir = '/expanse/lustre/projects/iit119/dcooper/MOR/systems'
output_dir = f'/expanse/lustre/projects/iit119/dcooper/MOR/final/resampled/analogues/{upper_limit}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'pdb'))
    os.mkdir(os.path.join(output_dir, 'dcd'))

for sim in sims:
    # Define outputs
    pdb_out = os.path.join(output_dir, 'pdb', "_".join(sim.split('_')[:-1]) + '.pdb')
    dcd_out = os.path.join(output_dir, 'dcd', "_".join(sim.split('_')[:-1]) + '.dcd')
    if not os.path.exists(dcd_out):
        print(sim, pdb_out, dcd_out)
        
        # Sim input
        dir = os.path.join(repexchange_dir, sim)
        pdb = os.path.join(pdb_dir, sim.split('_')[0] + '.pdb')
        
        # Make obj
        analysis = FultonMarketAnalysis(dir, pdb, skip=10, upper_limit=upper_limit, resids=np.array([84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]))
        
        # Importance Resampling
        try:
            analysis.importance_resampling(equilibration_method='PCA')
        except NameError:
            pass
        
        analysis.write_resampled_traj(pdb_out, dcd_out)