"""
USAGE >>>python UmbSam_EQUILIBRATION.py $INPUT_DIR $CENTROID_A_NAME $CENTROID_B_NAME $OUTPUT_DIR

Perform a Bilateral trailblazing


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
from FultonMarketUtils import *
from datetime import datetime
import mdtraj as md
import os, sys

#83.68 Joule/(mol * ang^2) = 20 cal/(mol * ang^2)

def get_some_objects(pdb_file, xml_file):
    pdb = PDBFile(pdb_file)
    with open(xml_file, 'r') as f:
        system = XmlSerializer.deserialize(f.read())
    openmm_top = pdb.topology
    mdtraj_top = md.Topology.from_openmm(openmm_top)
    return pdb, system, openmm_top, mdtraj_top


class Umbrella_Trailblazer():
    """
    """
    def __init__(self, input_dir, centroid_A_name, centroid_B_name, save_dir,
                 num_replicates=25, temp=367.447*unit.kelvin, pressure=1*unit.bar):
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
        self.sim_obs = {} # for A and B [PDBFile, System, openmm.Topology, mdtraj.Topology]
        for file_pair in (('A', centroid_A_pdb, centroid_A_xml), ('B', centroid_B_pdb, centroid_B_xml)):
            self.sim_obs[file_pair[0]] = get_some_objects(file_pair[1], file_pair[2])
        
        #Output
        self.stdout_fns = {'A': os.path.join(save_dir, 'trailblazing_A.stdout'), 'B': os.path.join(save_dir, 'trailblazing_B.stdout')}
        self.dcd_fns = {'A': os.path.join(save_dir, 'trailblazing_A.dcd'), 'B': os.path.join(save_dir, 'trailblazing_B.dcd')}
        self.final_pos_fns = {'A': os.path.join(save_dir, 'final_pos_A.dcd'), 'B': os.path.join(save_dir, 'final_pos_B.dcd')}
        for iden in ["A", "B"]:
            printf(f'Saving stdout to {self.stdout_fns[iden]}, \n dcd to {self.dcd_fns[iden]}, \n and final positions to {self.final_pos_fns[iden]}')
        
        #Determine for legs A and B how many frames have been ran
        self.n_frames_ran = {}
        for iden, dcd_fn in self.dcd_fns.items():
            if os.path.exists(dcd_fn):
                traj = md.load(dcd_fn, top=self.sim_obs[iden][3])
                self.n_frames_ran[iden] = traj.n_frames
                del traj
            else:
                self.n_frames_ran[iden] = 0
            printf(f'Found number of frames already on leg {iden} ran to be: {self.n_frames_ran[iden]}')
        
        # Simulation physical parameters
        self.num_replicates = num_replicates
        printf(f'Found number of replicates: {self.num_replicates}')
        
        self.temp = temp
        printf(f'Found simulation temperature: {temp}')
        
        self.pressure = pressure
        printf(f'Found simulation pressure: {self.pressure}')
        
        #Determine how many simulations to get halfway from A to B
        self.n_sims_per_centroid = self.num_replicates // 2
        self.is_middle_replicate = self.num_replicates % 2 == 1
        if self.is_middle_replicate:
            self.n_sims_per_centroid += 1
        
        # Set Final Pos and Box Vec
        self.final_poss, self.final_box_vecs = {}, {}
        for iden in ["A", "B"]:
            self.final_poss[iden] = np.empty((self.n_sims_per_centroid, self.sim_obs[iden][3].n_atoms, 3))
            self.final_box_vecs[iden] = np.empty((self.n_sims_per_centroid, 3, 3))
        printf(f"Shape of final positions and box_vectors: \n A: {self.final_poss['A'].shape} {self.final_box_vecs['A'].shape} \n B: {self.final_poss['B'].shape} {self.final_box_vecs['B'].shape} ")
        
    
    def assign_selection_string(self, intracellular_inds:dict):
        """
        Generates the selection string for restrained atoms
        Intracellular indices must be a 2-dict
            {"A": np.array,
             "B": np.array}
        """
        selection_strings = {}
        for iden in ["A", "B"]:
            resids = intracellular_inds[iden]
            restraint_selection_string = 'protein and ('
            for ind in resids:
                restraint_selection_string += f'(resid {ind}) or '
            restraint_selection_string = restraint_selection_string[:-4] + ')'
            selection_strings[iden] = restraint_selection_string
        
        self.restraint_selection_strings = selection_strings
    
    
    def interpolated_array_from_selections(self):
        """
        
        """
        pdbA, pdbB = self.pdb_fns["A"], self.pdb_fns["B"]
        selection_stringA, selection_stringB = self.restraint_selection_strings["A"], self.restraint_selection_strings["B"]
        
        traj1, traj2 = md.load(pdbA), md.load(pdbB)
        inds1, inds2 = traj1.top.select(selection_stringA), traj2.top.select(selection_stringB)
        assert inds1.shape == inds2.shape
        
        #traj2 = traj2.superpose(traj1, atom_indices=inds2, ref_atom_indices=inds1)
        xyz1, xyz2 = traj1.xyz[0, inds1], traj2.xyz[0, inds2]
        positions_array = np.empty((self.num_replicates, xyz1.shape[0], 3))
        
        lambdas = np.linspace(1, 0, self.num_replicates)
        gammas = 1 - lambdas
        for i in range(self.num_replicates):
            positions_array[i] = lambdas[i]*xyz1 + gammas[i]*xyz2
        
        return positions_array
    
    
    def assign_spring_attributes(self, intracellular_inds:dict, spring_constant=83.68*spring_constant_unit):
        """
        
        """
        
        self.assign_selection_string(intracellular_inds)
        self.spring_centers = self.interpolated_array_from_selections()
        self.spring_constant = spring_constant
        printf(f'Build spring centers with shape: {self.spring_centers.shape} and force constant {self.spring_constant}')
    
    
    def save_results(self):
        """
        
        """
        # Save final pos
        for iden in ["A", "B"]:
            traj = md.load(self.pdb_fns[iden])
            traj.xyz = copy.deepcopy(self.final_poss[iden])
            traj.unitcell_vectors = copy.deepcopy(self.final_box_vecs[iden])
            traj.save_dcd(self.final_pos_fns[iden])
            traj = md.load(self.final_pos_fns[iden], top=self.sim_obs[iden][3])
            traj.image_molecules()
            traj.save_dcd(self.final_pos_fns[iden])
            printf(f'Saved final positions to {self.final_pos_fns[iden]}')
    
    
    def run_leg(self, leg_iden, ts=2*unit.femtosecond, n_frames_per_replicate=2500, time_btw_frames=1*unit.picosecond):
        """
        leg iden must be in ["A", "B"]
        """
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
        init_positions = self.sim_obs[leg_iden][0].getPositions(asNumpy=True)
        if leg_iden == "A":
            spring_centers = self.spring_centers[:self.n_sims_per_centroid]
        elif leg_iden == "B":
            spring_centers = self.spring_centers[::-1][:self.n_sims_per_centroid]
            ##Also align onto trajA
            #pdbA, pdbB = self.pdb_fns["A"], self.pdb_fns["B"]
            #selection_stringA, selection_stringB = self.restraint_selection_strings["A"], self.restraint_selection_strings["B"]
            #
            #traj1, traj2 = md.load(pdbA), md.load(pdbB)
            #inds1, inds2 = traj1.top.select(selection_stringA), traj2.top.select(selection_stringB)
            #assert inds1.shape == inds2.shape
            #
            #traj2 = traj2.superpose(traj1, atom_indices=inds2, ref_atom_indices=inds1)
            #init_positions = traj2.openmm_positions(0)
        else:
            raise Exception("User Error: Failed to read docstring")
        
        # Reporter Parameters
        SDR_params = dict(file=self.stdout_fns[leg_iden], reportInterval=nstdout, step=True, time=True,
                          potentialEnergy=True, temperature=True, progress=False,
                          remainingTime=False, speed=True, volume=True,
                          totalSteps=n_frames_total, separator=' : ')
        DCDR_params = dict(file=self.dcd_fns[leg_iden], reportInterval=n_steps_per_frame, enforcePeriodicBox=True)
        
        for i, spring_center in enumerate(spring_centers):

            # If the dcd has these frames, just load
            if self.n_frames_ran[leg_iden] > i * n_frames_per_replicate:
                printf(f'Loading output for replicate {str(i)} ...')
                traj = md.load(self.dcd_fns[leg_iden], top=self.sim_obs[leg_iden][3])
                self.final_poss[leg_iden][i] = traj.xyz[i*n_frames_per_replicate]
                self.final_box_vecs[leg_iden][i] = traj.unitcell_vectors[i*n_frames_per_replicate]
                del traj

            else:            
                printf(f'Beginning simulation for replicate {str(i)}...')

                # Deep copy the init system, without the restraints or barostat, then add them
                system = copy.deepcopy(self.sim_obs[leg_iden][1])
                restrain_openmm_system_by_dsl(system, self.sim_obs[leg_iden][3],
                                              self.restraint_selection_strings[leg_iden],
                                              self.spring_constant, spring_center)
                system.addForce(MonteCarloBarostat(self.pressure, self.temp, 100))

                # Define the integrator and simulation etc.
                integrator = LangevinIntegrator(self.temp, 1/unit.picosecond, ts)
                simulation = Simulation(self.sim_obs[leg_iden][2], system, integrator)

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
                        traj = md.load(self.dcd_fns[leg_iden], top=self.sim_obs[leg_iden][3])
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
                self.n_frames_ran[leg_iden] += n_frames_per_replicate
                
                self.final_poss[leg_iden][i] = last_state.getPositions(asNumpy=True)._value
                self.final_box_vecs[leg_iden][i] = last_state.getPeriodicBoxVectors(asNumpy=True)._value
                

if __name__ == '__main__':
    assert len(sys.argv) == 5, "You may have the wrong number of arguments :("
    args = dict(input_dir = sys.argv[1],
                centroid_A_name = sys.argv[2],
                centroid_B_name = sys.argv[3],
                save_dir = sys.argv[4])
    
    UT = Umbrella_Trailblazer(**args)
    
    cb2_intracellular_inds = np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311])
    intra_inds_dict = {}

    for leg_iden in ["A", "B"]:
        if args[f"centroid_{leg_iden}_name"] in ["centroid_6", "centroid_13"]:
            #subtract 22 from these indices
            intra_inds_dict[leg_iden] = cb2_intracellular_inds - 22
        else:
            #subtract 23
            intra_inds_dict[leg_iden] = cb2_intracellular_inds - 23
            
    UT.assign_spring_attributes(intracellular_inds=intra_inds_dict)
    
    for leg_iden in ["A", "B"]:
        UT.run_leg(leg_iden)
        
    UT.save_results()
