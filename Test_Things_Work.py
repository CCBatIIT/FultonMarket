#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS
#WORK IN PROGRESS

# USAGE python Test_Things_Work.py
# Sequentially test each type of Fulton Market Simulation
# Skipping one does not take away a chance to test the next

#Test all imports for custom packages
from FultonMarket.FultonMarketUtils import *
from FultonMarket.Randolph import Randolph
from FultonMarket.FultonMarket import FultonMarket
from FultonMarket.FultonMarketPTwFR import FultonMarketPTwFR
from FultonMarket.FultonMarketUS import FultonMarketUS
from FultonMarket.Unilateral_Trailblazing import Unilateral_Umbrella_Trailblazer
from FultonMarket.Bilateral_Trailblazing import Bilateral_Umbrella_Trailblazer

#Other imports
import os, sys, glob
import openmm.unit as unit



def delete_all_files_in_dir(the_dir):
    files_wildcard = os.path.join(the_dir, '*')
    for f in glob.glob(files_wildcard):
        os.system(f'rm -r {f}')

cb2_intracellular_inds = np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311])
cb2_intracellular_inds = cb2_intracellular_inds - 23 # Zero indexed resids


#Fulton Market Test
response = input('Proceed with testing FultonMarket? y/n \n')
if response == 'y':
    #Setup Block
    test_output_dir = './Test_Cases/FM_test/'
    if not os.path.isdir(test_output_dir):
        os.mkdir(test_output_dir)
    else:
        response = input(f'Should delete the contents of {test_output_dir}? y/n \n')
        if response == 'y':
            delete_all_files_in_dir(test_output_dir)
    
    
    init_kwargs = dict(input_pdb='./Test_Cases/input/7OH.pdb',
                       input_system='./Test_Cases/input/7OH_sys.xml',
                       input_state='./Test_Cases/input/7OH_state.xml',
                       n_replicates=5)
    run_kwargs = dict(total_sim_time=1.0, iter_length=0.001, sim_length=0.1,
                      output_dir=test_output_dir, init_overlap_thresh=0.0, term_overlap_thresh=0.1)
    market = FultonMarket(**init_kwargs)
    market.run(**run_kwargs)


#PTwFR Test
response = input('Proceed with testing FultonMarketPtwFR? y/n \n')
if response == 'y':
    #Setup Block
    test_output_dir = './Test_Cases/FMPTwFR_test/'
    if not os.path.isdir(test_output_dir):
        os.mkdir(test_output_dir)
    else:
        response = input(f'Should delete the contents of {test_output_dir}? y/n \n')
        if response == 'y':
            delete_all_files_in_dir(test_output_dir)


    
    init_kwargs = dict(input_pdb='./Test_Cases/input/centroid_12.pdb',
                       input_system='./Test_Cases/input/centroid_12_sys.xml',
                       input_state=None,
                       n_replicates=5,
                       restrained_atoms_dsl=generate_selection_string(cb2_intracellular_inds),
                       T_min=310, T_max=315)
    run_kwargs = dict(total_sim_time=1.0, iter_length=0.001, sim_length=0.01,
                      output_dir=test_output_dir, init_overlap_thresh=0.0, term_overlap_thresh=0.1)
    market = FultonMarketPTwFR(**init_kwargs)
    market.run(**run_kwargs)


#Unilateral Trailblaze Test
response = input('Proceed with testing Unilateral Trailblzaing? y/n \n')
if response == 'y':
    #Setup Block
    test_output_dir = './Test_Cases/UniLat_test/'
    if not os.path.isdir(test_output_dir):
        os.mkdir(test_output_dir)
    else:
        response = input(f'Should delete the contents of {test_output_dir}? y/n \n')
        if response == 'y':
            delete_all_files_in_dir(test_output_dir)
    
    #Run Block
    UT = Unilateral_Umbrella_Trailblazer(input_dir='./Test_Cases/input/',
                                         centroid_A_name='centroid_12',
                                         centroid_B_name='centroid_17',
                                         save_dir=test_output_dir,
                                         num_replicates=5,
                                         temp=350*unit.kelvin,
                                         pressure=1*unit.bar)
    
    UT.assign_spring_attributes(intracellular_residue_indsA=cb2_intracellular_inds,
                                intracellular_residue_indsB=cb2_intracellular_inds,
                                spring_constant=83.68*spring_constant_unit)

    UT.run_trailblazing(ts=2*unit.femtosecond,
                        n_frames_per_replicate=25,
                        time_btw_frames=1*unit.picosecond)

    UT.save_results()


#US Test
response = input('Proceed with testing FultonMarketUS? y/n \n')
if response == 'y':
    
    #Setup Block
    test_output_dir = './Test_Cases/FMUS_test/'
    if not os.path.isdir(test_output_dir):
        os.mkdir(test_output_dir)
    else:
        response = input(f'Should delete the contents of {test_output_dir}? y/n \n')
        if response == 'y':
            delete_all_files_in_dir(test_output_dir)
            
    init_kwargs = dict(input_pdb=['./Test_Cases/input/centroid_12.pdb', './Test_Cases/input/centroid_17.pdb'],
                       input_system='./Test_Cases/input/centroid_12_sys.xml',
                       restrained_atoms_dsl=generate_selection_string(cb2_intracellular_inds),
                       init_positions_dcd='./Test_Cases/UniLat_test/final_pos.dcd',
                       n_replicates=5)
    run_kwargs = dict(total_sim_time=1.0, iter_length=0.001, sim_length=0.1,
                      output_dir=test_output_dir, init_overlap_thresh=0.0, term_overlap_thresh=0.1)
    market = FultonMarketUS(**init_kwargs)
    market.run(**run_kwargs)


    
