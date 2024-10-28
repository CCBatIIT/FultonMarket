"""
USAGE >>>python UmbSam_EQUILIBRATION.py $INPUT_DIR $CENTROID_A_NAME $CENTROID_B_NAME $OUTPUT_DIR

PARAMETERS:
-----------
    INPUT_DIR: Absolute path to directory with .pdb and _sys.xml input for centroids A and B
    CENTROID_A/B_NAME: Name of centroid (e.g. centroid_1)
    OUTPUT_DIR: Absolute path to save output
"""


from openmm import *
import numpy as np
from openmm.app import *
import openmm.unit as unit
from FultonMarket import FultonMarket
from FultonMarketUtils import *
from datetime import datetime
import mdtraj as md
import os, sys


# User inputs
input_dir = sys.argv[1]
assert os.path.exists(input_dir)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found input directory:', input_dir, flush=True)

centroid_A_name = sys.argv[2]
centroid_A_pdb = os.path.join(input_dir, centroid_A_name + '.pdb')
centroid_A_xml = os.path.join(input_dir, centroid_A_name + '_sys.xml')
assert os.path.exists(centroid_A_pdb) and os.path.exists(centroid_A_xml)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found Centroid A:', centroid_A_name, flush=True)

centroid_B_name = sys.argv[3]
centroid_B_pdb = os.path.join(input_dir, centroid_B_name + '.pdb')
centroid_B_xml = os.path.join(input_dir, centroid_B_name + '_sys.xml')
assert os.path.exists(centroid_B_pdb) and os.path.exists(centroid_B_xml)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found Centroid B:', centroid_B_name, flush=True)

save_dir = sys.argv[4]
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
assert os.path.exists(save_dir)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found output directory:', save_dir, flush=True)


# Open input 
pdb = PDBFile(centroid_A_pdb)
with open(centroid_A_xml, 'r') as f:
    system_A_init = XmlSerializer.deserialize(f.read())
openmm_topology = pdb.topology
mdtraj_topology = md.Topology.from_openmm(openmm_topology)


# Ouput
stdout_fn = os.path.join(save_dir, 'trailblazing.stdout')
dcd_fn = os.path.join(save_dir, 'trailblazing.dcd')
final_pos_fn = os.path.join(save_dir, 'final_pos.dcd')
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saving stdout, dcd, and final positions to:', stdout_fn, dcd_fn, final_pos_fn, flush=True)
if os.path.exists(dcd_fn):
    traj = md.load(dcd_fn, top=centroid_A_pdb)
    n_frames_ran = traj.n_frames
else:
    n_frames_ran = 0
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found number of frames already ran to be:', n_frames_ran, flush=True)


#Assign the restraints to the thermodynamic states
restraint_selection_string = 'protein and ('
intracellular_inds = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]
intracellular_inds = [ind - 67 for ind in intracellular_inds]
for ind in intracellular_inds:
    restraint_selection_string += f'(resid {ind}) or '
restraint_selection_string = restraint_selection_string[:-4] + ')'


# Simulation physical parameters
num_replicates = 25
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found number of replicates:', num_replicates, flush=True)
temp_min, temp_max = 300*unit.kelvin, 367.447*unit.kelvin
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found temperature range of:', temp_min, temp_max, flush=True)
pressure = 1*unit.bar
spring_constant = 83.68*spring_constant_unit #20 cal/(mol * ang^2)
spring_centers = make_interpolated_positions_array(centroid_A_pdb, centroid_B_pdb, num_replicates)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Build spring centers with shape:', spring_centers.shape, flush=True)



# Simulation length parameters
ts = 2*unit.femtosecond
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found timestep of:', ts, flush=True)
n_frames_per_replicate = 2500
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found no. of frames per replicate:', n_frames_per_replicate, flush=True)
time_btw_frames = 1*unit.picosecond 
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found time between frames:', time_btw_frames, flush=True)
n_steps_per_frame = round(time_btw_frames / ts)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found steps per frame:', n_steps_per_frame, flush=True)
nstdout = n_steps_per_frame // 5
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found nstdout of:', nstdout, flush=True)
n_frames_total = n_frames_per_replicate * num_replicates
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Found total no. of frames:', n_frames_total, flush=True)


# Input positions
init_positions = pdb.getPositions(asNumpy=True)


# Reporter Parameters
SDR_params = dict(file=stdout_fn, reportInterval=nstdout, step=True, time=True,
                  potentialEnergy=True, temperature=True, progress=False,
                  remainingTime=False, speed=True, volume=True,
                  totalSteps=n_frames_total, separator=' : ')
DCDR_params = dict(file=dcd_fn, reportInterval=n_steps_per_frame, enforcePeriodicBox=True)


# Iterate through spring centers and simulate
final_pos = np.empty((len(spring_centers), init_positions.shape[0], 3))
final_box_vec = np.empty((len(spring_centers), 3, 3))

nan_counter = 0
while (n_frames_ran < n_frames_per_replicate * num_replicates) and nan_counter < 5:
    for i, spring_center in enumerate(spring_centers):

        # If the dcd has these frames, just load
        if n_frames_ran > i * n_frames_per_replicate:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Loading output for replicate', str(i) + '...', flush=True)
            final_pos[i] = traj.xyz[i*n_frames_per_replicate]
            final_box_vec[i] = traj.unitcell_vectors[i*n_frames_per_replicate]

        else:            
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Beginning simulation for replicate', str(i) + '...', flush=True)

            # Deep copy the init system, without the restraints or barostat, then add them
            system = copy.deepcopy(system_A_init)
            restrain_openmm_system_by_dsl(system, mdtraj_topology, restraint_selection_string, spring_constant, spring_center)
            system.addForce(MonteCarloBarostat(pressure, temp_max, 100))
            
            # Define the integrator and simulation etc.
            integrator = LangevinIntegrator(temp_max, 1/unit.picosecond, ts)
            simulation = Simulation(openmm_topology, system, integrator)
        
            # First simulation checks, set positions and start reporters differently
            if 'last_state' in locals():  
                init_positions = last_state.getPositions(asNumpy=True)
                simulation.context.setPositions(init_positions)
                simulation.context.setVelocities(last_state.getVelocities())
                append = True
                
            else:
                if i == 0:
                    append = False
                    
                else:
                    temp_pdb = os.path.join(os.getcwd(), 'temp.pdb')
                    traj[-1].save_pdb(temp_pdb)
                    pdb = PDBFile(temp_pdb)
                    init_positions = pdb.getPositions(asNumpy=True)
                    append = True
                    
                simulation.context.setPositions(init_positions)
                simulation.context.setVelocitiesToTemperature(temp_max)

            for param_set in [SDR_params, DCDR_params]:
                    param_set['append'] = append
            
            # Init and append reporters
            SDR = StateDataReporter(**SDR_params)
            DCDR = DCDReporter(**DCDR_params)
            simulation.reporters.append(SDR)
            simulation.reporters.append(DCDR)
        
            # Run that bish ---> PAUSE -DC
            start = datetime.now()
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Minimizing...', flush=True)
            simulation.minimizeEnergy() 
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Minimizing finished', flush=True)
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Taking', n_steps_per_frame * n_frames_per_replicate, 'steps', flush=True)
            
            # Try to take steps, if not interpolate and start for loop again
            try:                
                simulation.step(n_steps_per_frame * n_frames_per_replicate)
                
            except:
                new_spring_centers = np.mean((spring_centers[i], spring_centers[i-1]), axis=0)
                spring_centers = np.insert(spring_centers, i, new_spring_centers, axis=0)
                final_pos = np.insert(final_pos, i, np.empty((init_positions.shape[0], 3)), axis=0)
                final_box_vec = np.insert(final_box_vec, i, np.empty((3, 3)), axis=0)
                traj = md.load(dcd_fn, top=centroid_A_pdb)
                num_replicates += 1
                nan_counter += 1
                
                break 
                
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Time to simulate replicate', i, 'for', n_frames_per_replicate * time_btw_frames, 'took', datetime.now() - start, flush=True)
        
            # After the run, we need the state to set the next positions
            last_state = simulation.context.getState(getPositions=True, getVelocities=True)
            final_pos[i] = last_state.getPositions(asNumpy=True)._value
            box_vec = last_state.getPeriodicBoxVectors(asNumpy=True)
            final_box_vec[i] = last_state.getPeriodicBoxVectors(asNumpy=True)._value
            n_frames_ran += n_frames_per_replicate
            nan_counter = 0

        print(n_frames_ran)

# Catch error
if nan_counter >= 5:
    raise Exception('nan_counter')

# Save final pos
traj = md.load_pdb(centroid_A_pdb)
traj.xyz = copy.deepcopy(final_pos)
traj.unitcell_vectors = copy.deepcopy(final_box_vec)
traj.save_dcd(final_pos_fn)
traj = md.load(final_pos_fn, top=centroid_A_pdb)
traj.image_molecules()
traj.save_dcd(final_pos_fn)
print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Saved final positions to', final_pos_fn, flush=True)

