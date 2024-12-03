"""
USAGE: python RUN_FULTONMARKETUS.py $PDB/XML_INPUT_DIR $DCD_INPUT_DIR $NAME1 $NAME2 $OUTPUT_DIR $SIM_TIME $NUM_OF_REPLICA

PARAMETERS:
-----------
    Required
    INPUT_DIR: absolute path to the directory with input xml and pdb
    DCD_INPUT_DIR: absolute path to the directory containing the trailblazing final positions
    NAME1 & NAME2: file name before extensions for the two centroids
    OUTPUT_DIR: absolute path to the directory where a subdirectory with output will be stored

    Optional
    SIM_TIME: Total simulation aggregate time. Default is 500 ns.
    NUM_OF_REPLICA: number of replica to start with between T_min (300 K) and T_max (360 K)
"""


import os, sys, math
import numpy as np
from FultonMarket.FultonMarketUS import FultonMarketUS

# Inputs
input_dir = sys.argv[1]
dcd_dir = sys.argv[2]
name1 = sys.argv[3]
name2 = sys.argv[4]
input_sys = os.path.join(input_dir, name1+'_sys.xml')
input_pdb = [os.path.join(input_dir, name1+'.pdb'), os.path.join(input_dir, name2+'.pdb')]
input_dcd = os.path.join(dcd_dir, 'final_pos.dcd')
print(input_pdb)
print(input_dcd)

# Outputs
output_dir = os.path.join(sys.argv[5], name1 + '_' + name2)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert os.path.exists(output_dir)

# Simulation parameters
try:
    total_sim_time = int(sys.argv[6])
except:
    total_sim_time = 500

try:
    n_replicates = int(sys.argv[7])
except:
    n_replicates = 25

# Restraints
selection_string = 'protein and ('
cb2_intracellular_inds = np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311])
cb2_intracellular_inds -= 23
for ind in cb2_intracellular_inds:
    selection_string += f'(resid {ind}) or '
selection_string = selection_string[:-4] + ')'

# Run rep exchange
market = FultonMarketUS(input_pdb=input_pdb, input_system=input_sys, init_positions_dcd=input_dcd, n_replicates=n_replicates, restrained_atoms_dsl=selection_string)

# RUN
market.run(total_sim_time=total_sim_time, iter_length=0.001, sim_length=5.0, output_dir=output_dir)

