import os, sys
sys.path.append('FultonMarket')
from openmm import *
import numpy as np
from openmm.app import *
import openmm.unit as unit
from .FultonMarketUtils import *
from datetime import datetime
import mdtraj as md


#83.68 Joule/(mol * ang^2) = 20 cal/(mol * ang^2)


class Unilateral_Umbrella_Trailblazer():
    """
    
    Perform a Unilateral trailblazing
    
    
    PARAMETERS:
    -----------
        INPUT_DIR: Absolute path to directory with .pdb and _sys.xml input for centroids A and B
        CENTROID_A/B_NAME: Name of centroid (e.g. centroid_1)
        OUTPUT_DIR: Absolute path to save output
    """
    def __init__(self, input_dir, centroid_A_name, centroid_B_name, save_dir,
                 num_replicates=5, temp=367.447*unit.kelvin, pressure=1*unit.bar): #REMOVE 5 -> 25
        """
        """
        # User inputs
        assert os.path.exists(input_dir)
        printf(f'Found input directory: {input_dir}')

        centroid_A_pdb = os.path.join(input_dir, centroid_A_name + '.pdb')
        centroid_A_xml = os.path.join(input_dir, centroid_A_name + '_sys.xml')
        assert os.path.exists(centroid_A_pdb) and os.path.exists(centroid_A_xml)
        printf(f'Found Centroid A: {centroid_A_name}')

        centroid_B_pdb = os.path.join(input_dir, centroid_B_name + '.pdb')
        centroid_B_xml = os.path.join(input_dir, centroid_B_name + '_sys.xml')
        assert os.path.exists(centroid_B_pdb) and os.path.exists(centroid_B_xml)
        printf(f'Found Centroid B: {centroid_B_name}')
        
        self.pdb_fns = {"A": centroid_A_pdb, "B": centroid_B_pdb}

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        assert os.path.exists(save_dir)
        printf(f'Found output directory: {save_dir}')
        self.save_dir = save_dir
        
        # Open input 
        self.sim_obs = get_some_objects(centroid_A_pdb, centroid_A_xml)
        #Sim obs has shape (PDBFile, openmm.System, openmm.Topology, mdtraj.Topology)
                
        #Output
        self.stdout_fn = os.path.join(save_dir, 'trailblazing.stdout')
        self.dcd_fn = os.path.join(save_dir, 'trailblazing.dcd')
        self.final_pos_fn = os.path.join(save_dir, 'final_pos.dcd')
        printf(f'Saving stdout to {self.stdout_fn}, \n dcd to {self.dcd_fn}, \n and final positions to {self.final_pos_fn}')
        
        #Determine for legs A and B how many frames have been ran
        if os.path.exists(self.dcd_fn):
            traj = md.load(self.dcd_fn, top=self.sim_obs[3])
            self.n_frames_ran = traj.n_frames
            del traj
        else:
            self.n_frames_ran = 0
        printf(f'Found number of frames already on ran to be: {self.n_frames_ran}')
        
        # Simulation physical parameters
        self.num_replicates = num_replicates
        printf(f'Found number of replicates: {self.num_replicates}')
        
        self.temp = temp
        printf(f'Found simulation temperature: {temp}')
        
        self.pressure = pressure
        printf(f'Found simulation pressure: {self.pressure}')

        self.final_pos = np.empty((self.num_replicates, self.sim_obs[3].n_atoms, 3))
        self.final_box_vecs = np.empty((self.num_replicates, 3, 3))
        printf(f"Shape of final positions and box_vectors: {self.final_pos.shape} & {self.final_box_vecs.shape}")


    def assign_spring_attributes(self, intracellular_residue_indsA:np.array,
                                 intracellular_residue_indsB:np.array=None,
                                 spring_constant=83.68*spring_constant_unit):
        """
        
        """
        #If resids are not provided for B assume they are the same as A
        if intracellular_residue_indsB is None:
            intracellular_residue_indsB = intracellular_residue_indsA.copy()
        #Get selection strings for all atoms in the given resids lists
        selection_stringA = generate_selection_string(intracellular_residue_indsA)
        selection_stringB = generate_selection_string(intracellular_residue_indsA)
        
        my_args = dict(spring_centers1_pdb=self.pdb_fns["A"],
                       spring_centers2_pdb=self.pdb_fns["B"],
                       selection_1=selection_stringA,
                       selection_2=selection_stringB,
                       num_replicates=self.num_replicates)

        self.restraint_selection_string = selection_stringA
        self.spring_centers, indsA, indsB = make_interpolated_positions_array_from_selections(**my_args)
        self.spring_constant = spring_constant
        printf(f'Build spring centers with shape: {self.spring_centers.shape} and force constant {self.spring_constant}')
    
    
    def save_results(self):
        """
        
        """
        # Save final pos
        traj = md.load(self.pdb_fns["A"])
        traj.xyz = copy.deepcopy(self.final_pos)
        traj.unitcell_vectors = copy.deepcopy(self.final_box_vecs)
        traj.save_dcd(self.final_pos_fn)
        traj = md.load(self.final_pos_fn, top=self.sim_obs[3])
        traj.image_molecules()
        traj.save_dcd(self.final_pos_fn)
        printf(f'Saved final positions to {self.final_pos_fn}')
    
    
    def run_trailblazing(self, ts=2*unit.femtosecond,
                         n_frames_per_replicate=2500,
                         time_btw_frames=1*unit.picosecond):
        
        # Simulation length parameters
        n_steps_per_frame = round(time_btw_frames / ts)
        nstdout = n_steps_per_frame // 5
        n_frames_total = n_frames_per_replicate * self.num_replicates

        printf(f'Found timestep of: {ts}')
        printf(f'Found no. of frames per replicate: {n_frames_per_replicate}')
        printf(f'Found time between frames: {time_btw_frames}')
        printf(f'Found steps per frame: {n_steps_per_frame}')
        printf(f'Found nstdout of: {nstdout}')
        printf(f'Found total no. of frames: {n_frames_total}')
        
        # Input positions
        init_positions = self.sim_obs[0].getPositions(asNumpy=True)
        
        # Reporter Parameters
        SDR_params = dict(file=self.stdout_fn, reportInterval=nstdout, step=True, time=True,
                          potentialEnergy=True, temperature=True, progress=False,
                          remainingTime=False, speed=True, volume=True,
                          totalSteps=n_frames_total, separator=' : ')
        DCDR_params = dict(file=self.dcd_fn, reportInterval=n_steps_per_frame, enforcePeriodicBox=True)
        
        for i, spring_center in enumerate(self.spring_centers):
            # If the dcd has these frames, just load
            if self.n_frames_ran > i * n_frames_per_replicate:
                printf(f'Loading output for replicate {str(i)} ...')
                traj = md.load(self.dcd_fn, top=self.sim_obs[3])
                self.final_pos[i] = traj.xyz[i*n_frames_per_replicate]
                self.final_box_vecs[i] = traj.unitcell_vectors[i*n_frames_per_replicate]
                del traj
            else:            
                printf(f'Beginning simulation for replicate {str(i)}...')
                # Deep copy the init system ("A"), without the restraints or barostat, then add them
                system = copy.deepcopy(self.sim_obs[1])
                restrain_openmm_system_by_dsl(system, self.sim_obs[3],
                                              self.restraint_selection_string,
                                              self.spring_constant, spring_center)
                system.addForce(MonteCarloBarostat(self.pressure, self.temp, 100))

                # Define the integrator and simulation etc.
                integrator = LangevinIntegrator(self.temp, 1/unit.picosecond, ts)
                simulation = Simulation(self.sim_obs[2], system, integrator)

                # First simulation checks, set positions and start reporters differently
                if 'last_state' in locals():
                    init_positions = last_state.getPositions(asNumpy=True)
                    simulation.context.setPositions(last_state.getPositions())
                    simulation.context.setPeriodicBoxVectors(*last_state.getPeriodicBoxVectors())
                    simulation.context.setVelocities(last_state.getVelocities())
                    append = True

                else:
                    if i == 0:
                        append = False
                    else:
                        temp_pdb = os.path.join(os.getcwd(), 'temp.pdb')
                        traj = md.load(self.dcd_fn, top=self.sim_obs[3])
                        traj[-1].save_pdb(temp_pdb)
                        pdb = PDBFile(temp_pdb)
                        os.remove(temp_pdb)
                        init_positions = pdb.getPositions(asNumpy=True)
                        init_box_vec = [Vec3(*vector) for vector in traj.unitcell_vectors[-1]]
                        simulation.context.setPeriodicBoxVectors(*init_box_vec)
                        del traj, pdb, init_box_vec
                        append = True

                    simulation.context.setPositions(init_positions)
                    simulation.context.setVelocitiesToTemperature(self.temp)

                for param_set in [SDR_params, DCDR_params]:
                    param_set['append'] = append

                # Init and append reporters
                SDR = StateDataReporter(**SDR_params)
                DCDR = DCDReporter(**DCDR_params)
                simulation.reporters.append(SDR)
                simulation.reporters.append(DCDR)

                # Run that bish ---> PAUSE -DC
                start = datetime.now()
                printf('Minimizing...')
                simulation.minimizeEnergy() 
                printf('Minimizing finished')
                printf(f'Taking {n_steps_per_frame * n_frames_per_replicate} steps...')
                simulation.step(n_steps_per_frame * n_frames_per_replicate)
                printf(f'Time to simulate replicate {i} for {n_frames_per_replicate * time_btw_frames} took {datetime.now() - start}')

                # After the run, we need the state to set the next positions
                last_state = simulation.context.getState(getPositions=True, getVelocities=True)
                box_vec = last_state.getPeriodicBoxVectors(asNumpy=True)
                self.n_frames_ran += n_frames_per_replicate
                
                self.final_pos[i] = last_state.getPositions(asNumpy=True)._value
                self.final_box_vecs[i] = last_state.getPeriodicBoxVectors(asNumpy=True)._value
                