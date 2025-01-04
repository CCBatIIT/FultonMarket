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


import os, sys, argparse
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_dir', help="absolute path to the directory with input xml and pdb")
parser.add_argument('name', help="pdb file before the extension")
parser.add_argument('output_dir', help="absolute path to the directory where a subdirectory with output will be stored")
parser.add_argument('replicate', help="replicate number of simulation. THIS MUST BE SPECIFIED to avoid accidental overwritting")
parser.add_argument('-t', '--total-sim-time', default=500, help="total simulation aggregate time. Default is 500 ns.", type=int)
parser.add_argument('-s', '--sub-sim-length', default=50, help="time of sub simulations. Default is 50 ns. Recommended this is smaller for larger systems.", type=int)
parser.add_argument('-n', '--n-replica', default=100, help="number of replica to start with between T_min (300 K) and T_max (360 K)", type=int)
args = parser.parse_args()

sys.path.append('FultonMarket')
from FultonMarket import FultonMarket

if __name__ == '__main__':

    # Inputs
    input_dir = args.input_dir
    name = args.name
    input_sys = os.path.join(input_dir, name+'_sys.xml')
    input_state = os.path.join(input_dir, name+'_state.xml')
    input_pdb = os.path.join(input_dir, name+'.pdb')
    
    # Outputs
    output_dir = os.path.join(sys.argv[3], name + '_' + str(args.replicate))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    assert os.path.exists(output_dir)

    
    # Run rep exchange
    market = FultonMarket(input_pdb=input_pdb, input_system=input_sys, input_state=input_state, n_replicates=args.n_replica)
    
    market.run(total_sim_time=args.total_sim_time, iter_length=0.001, sim_length=args.sub_sim_length, output_dir=output_dir)

