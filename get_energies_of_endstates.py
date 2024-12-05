import os, sys, glob
import numpy as np
from FultonMarket.analysis.FultonMarketAnalysis import FultonMarketAnalysis

resids = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311]
resids = [resid - 23 for resid in resids]


centroid_dir = sys.argv[1]
input_pdb = sys.argv[2]
name = input_pdb.split('/')[-1].replace('.pdb', '')
save_dir = sys.argv[3]

al_norest = FultonMarketAnalysis(centroid_dir, input_pdb, skip=10, resids=resids, remove_harmonic=True)
al_rest = FultonMarketAnalysis(centroid_dir, input_pdb, skip=10, resids=resids, remove_harmonic=False)


for al in [al_norest, al_rest]:
    if not hasattr(al, 't0'):
        al.equilibration_method = 'energy'
        al._determine_equilibration()

eners = [al_norest.get_state_energies(0)[al_norest.t0:],
         al_norest.get_state_energies(-1)[al_norest.t0:],
         al_rest.get_state_energies(0)[al_rest.t0:],
         al_rest.get_state_energies(-1)[al_rest.t0:]]
names = [f'{name}_state0_norest', f'{name}_state1_norest', f'{name}_state0_rest', f'{name}_state1_rest']

for ener_arr, name in zip(eners, names):
    np.save(os.path.join(save_dir, name)+'.npy', ener_arr)

