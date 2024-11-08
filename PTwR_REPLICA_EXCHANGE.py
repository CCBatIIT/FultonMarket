"""
USAGE: python REPLICA_EXCHANGE.py $INPUT_DIR $NAME $OUTPUT_DIR $REPLICATE $SIM_TIME $NUM_OF_REPLICA 

PARAMETERS:
-----------
    INPUT_DIR: absolute path to the directory with input xml and pdb
    NAME: pdb file before the extension
    OUTPUT_DIR: absolute path to the directory where a subdirectory with output will be stored
    REPLICATE: Replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting
    SIM_TIME: Total simulation aggregate time. Default is 500 ns. 
    NUM_OF_REPLICA: number of replica to start with between T_min (300 K) and T_max (360 K)
"""


import os, sys
sys.path.append('FultonMarket')
from FultonMarket import FultonMarket

# Inputs
input_dir = sys.argv[1]
name = sys.argv[2]
input_sys = os.path.join(input_dir, name+'_sys.xml')
input_state = os.path.join(input_dir, name+'_state.xml')
input_pdb = os.path.join(input_dir, name+'.pdb')
print(input_pdb)

# Outputs
rep = int(sys.argv[4])
output_dir = os.path.join(sys.argv[3], name + '_' + str(rep))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert os.path.exists(output_dir)

# Simulation parameters
try:
    total_sim_time = int(sys.argv[5])
except:
    total_sim_time = 500

sub_sim_length = 0.1

try:
    n_replica = int(sys.argv[6])
except:
    n_replica = 95

# Restraints
selection_string = 'protein and ('
intracellular_inds = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311]
intracellular_inds = [ind - 23 for ind in intracellular_inds]
print(len(intracellular_inds))
for ind in intracellular_inds:
    selection_string += f'(resid {ind}) or '
selection_string = selection_string[:-4] + ')'

# Run rep exchange
market = FultonMarket(input_pdb=input_pdb, input_system=input_sys, input_state=None)

market.run(total_sim_time=total_sim_time, iteration_length=0.001,
	   n_replicates=n_replica, sim_length=sub_sim_length,
           output_dir=output_dir, restrained_atoms_dsl=selection_string)

