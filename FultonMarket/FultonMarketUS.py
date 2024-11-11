#Package Imports
from openmm import *
from openmm.app import *
from openmmtools.utils.utils import TrackedQuantity
from openmm.unit.quantity import Quantity
from openmm.vec3 import Vec3
import mdtraj as md
import os, faulthandler
import numpy as np
from typing import List
from copy import deepcopy

#Custom Imports
from .FultonMarketUtils import *
from .FultonMarketPTwFR import FultonMarketPTwFR
from .analysis.FultonMarketAnalysis import FultonMarketAnalysis

#Set some things
np.seterr(divide='ignore', invalid='ignore')
faulthandler.enable()


class FultonMarketUS(FultonMarketPTwFR):
    """
    Umbrella Sampling at a set temperature

    Methods Custom to This Class:
    
        __init__(self, input_pdb: List[str], input_system: List[str], restrained_atoms_dsl: str,
                 init_positions_dcd: List[str], K=83.68*spring_constant_unit, T_min: float=310,
                 T_max: float=310, n_replicates: int=12)
        
        _set_init_from_trailblazing(self)
        
    Overwrites to Inherited Methods:

        run(self, total_sim_time: float, iter_length: float=0.01, dt: float=2.0, 
            sim_length=5, init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/')

                This method was overwritten to switch the default sim_length (Randolph Length) to 5 agg nanoseconds
                Between Randolph runs, the resampling of positions is also added.
                A Default iteration length of 10 picoseconds is also set specifically for this class
    
        _get_restrained_atoms(self)
        
        _set_init_positions(self)
        
        _set_init_box_vectors(self)
        
    Methods Inherited from FultonMarketPTwFR:    
    
        _set_parameters(self)
        
        _build_states(self)
        
        _build_sampler_states(self)
        
        _build_thermodynamic_states(self)
        
        _save_sub_simulation(self)
        
        _load_initial_args(self)

    Methods inherited from FultonMarket through FultonMarketPTwFR:
    
        run(self, total_sim_time: float, iter_length: float, dt: float=2.0, sim_length=50,
            init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35, output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/'))
        
        _configure_experiment_parameters(self, sim_length=50)
        
        _recover_arguments(self)

    """

    def __init__(self, 
                 input_pdb: List[str], 
                 input_system: List[str], 
                 restrained_atoms_dsl: str,
                 init_positions_dcd: List[str], 
                 K=83.68*spring_constant_unit,
                 T_min: float=310, 
                 T_max: float=310, 
                 n_replicates: int=12):
        """
        Initialize a Fulton Market obj. 

        Parameters:
        -----------
            input_pdb (str):
                String path to pdb to run simulation. 

            input_system (str):
                String path to OpenMM system (.xml extension) file that contains parameters for simulation. 

            input_state (str):
                String path to OpenMM state (.xml extension) file that contains state for reference. 


        Returns:
        --------
            FultonMarket obj.
        """
        printf(f'Welcome to FultonMarketUS.')

        # Copy input_pdb
        self.input_pdb_copy = deepcopy(input_pdb)
        
        super().__init__(input_pdb=input_pdb[0],
                         input_system=input_system,
                         restrained_atoms_dsl=restrained_atoms_dsl,
                         K=K,
                         T_min=T_min,
                         T_max=T_max,
                         n_replicates=n_replicates)
        printf("Finished Initializing Fulton Market PT with Restraints")

        # Set attr
        self.init_positions_dcd = init_positions_dcd

        # Set init_positons
        self._set_init_from_trailblazing()



    def _set_init_from_trailblazing(self):

        # Make assertions
        printf(f"Assigning Initial Positions from Bilateral Trailblazing DCDs {self.init_positions_dcd}")
        assert False not in [os.path.exists(dcd_fn) for dcd_fn in self.init_positions_dcd], self.init_positions_dcd

        # Load traj objs
        init_traj1 = md.load(self.init_positions_dcd[0], top=self.input_pdb[0])
        init_traj2 = md.load(self.init_positions_dcd[1], top=self.input_pdb[1])
        assert init_traj1.n_frames == init_traj2.n_frames

        # Replace env in init_traj2 if different
        init_traj2 = swap_traj_env(init_traj1, init_traj2)

        num_per_leg = self.n_replicates // 2
        
        #In the bilateral case, feed the initial positions for the second half backward (reversed)
        self.B_range = deepcopy(num_per_leg)
        if self.n_replicates %2 == 1:
            assert num_per_leg + 1 == init_traj1.n_frames
            self.A_range = num_per_leg + 1

        else:
            assert num_per_leg == init_traj1.n_frames
            self.A_range = num_per_leg 

        init_traj = init_traj1[:self.A_range].join(init_traj2[:self.B_range][::-1])
        # init_traj[0].save_pdb('init_traj.pdb')
        # init_traj.save_dcd('init_traj.dcd')
        print('init_traj', init_traj.xyz.shape)

        
        # Get pos, box_vectors
        self.init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_traj.xyz, mask=False, fill_value=1e+20), unit=unit.nanometer))
        self.init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_traj.unitcell_vectors, mask=False, fill_value=1e+20), unit=unit.nanometer))
        

    def run(self, 
            total_sim_time: float, 
            iter_length:float=0.01, 
            dt: float=2.0, 
            sim_length=5,
            init_overlap_thresh: float=0.5, 
            term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/')):

        # Set attr
        self.total_sim_time = total_sim_time
        self.iter_length = iter_length
        self.dt = dt
        self.sim_length = sim_length
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        # Prepare output
        self.output_dir = output_dir
        self.output_ncdf = os.path.join(self.output_dir, 'output.ncdf')
        self.checkpoint_ncdf = os.path.join(self.output_dir, 'output_checkpoint.ncdf')
        self.save_dir = os.path.join(self.output_dir, 'saved_variables')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        printf(f'Found total simulation time of {self.total_sim_time} nanoseconds')
        printf(f'Found iteration length of {self.iter_length} nanoseconds')
        printf(f'Found timestep of {self.dt} femtoseconds')
        printf(f'Found number of replicates {self.n_replicates}')
        printf(f'Found initial acceptance rate threshold {self.init_overlap_thresh}')
        printf(f'Found terminal acceptance rate threshold {self.term_overlap_thresh}')
        printf(f'Found output_dir {self.output_dir}')
        printf(f'Found Temperature Schedule {[np.round(T._value, 1) for T in self.temperatures]} Kelvin')
            

        # Loop through short 50 ns simulations to allow for .ncdf truncation
        self._configure_experiment_parameters(sim_length=self.sim_length)
        while self.sim_no < self.total_n_sims:
                         
            # Initialize Randolph
            if self.sim_no > 0:
                self._load_initial_args() #sets positions, velocities, box_vecs, temperatures, and spring_constants

            # Build states
            self._build_states()

            # Set parameters
            self._set_parameters()

            self.simulation = Randolph(**self.params)
            
            # Run simulation
            self.simulation.main(init_overlap_thresh=init_overlap_thresh, term_overlap_thresh=term_overlap_thresh)

            # Save simulation
            self._save_sub_simulation()

            #Resample from the saved directories to get new positions for each state
            self._resample_init_positions()

            # Delete output.ncdf files if not last simulation 
            if not self.sim_no+1 == self.total_n_sims:
                os.remove(self.output_ncdf)
                os.remove(self.checkpoint_ncdf)

            # Update counter
            self.sim_no += 1
    
    
    
    
    def _set_init_positions(self):
        pass



    def _set_init_box_vectors(self):
        pass
    

    def _get_restrained_atoms(self):
        """
        """

        # Reset self.input_pdb to List
        self.input_pdb = deepcopy(self.input_pdb_copy)

        # Open mdtraj obj
        traj = md.load(self.input_pdb[0])
        mdtraj_topology = traj.topology
    
        # Get indices of the atoms to restrain
        self.restrained_atom_indices = mdtraj_topology.select(self.restrained_atoms_dsl)
        if len(self.restrained_atom_indices) == 0:
            raise Exception('No Atoms To Restrain!')

        # Get spring centers
        self.spring_centers = make_interpolated_positions_array(self.input_pdb[0], self.input_pdb[1], self.n_replicates)
        assert len(self.temperatures) == self.spring_centers.shape[0]
        printf('Restraining each state to the unique positions of provided selection string')


    def _resample_init_positions(self):
        """
        """
        #Load an analyzer
        input_dir = os.path.abspath(os.path.join(self.save_dir, '..'))
        #The analyzer will handle the loading of energies and any backfilling
        analyzer = FultonMarketAnalysis(input_dir, self.input_pdb)
        #Set the new intial positions and box vecs by resampling
        new_init_positions = []
        new_init_box_vecs = []
        for i in range(self.simulation.n_replicates):
            analyzer.importance_resampling(n_samples=1, equilibration_method='None', specify_state=i)
            #sets analyzer.resampled_inds and analyzer.weights
            traj = analyzer.write_resampled_traj('temp.pdb', 'temp.dcd', return_traj=True)
            #clean up from that line
            os.remove('temp.pdb')
            os.remove('temp.dcd')
            #Add positions and box vectors to the list
            new_init_positions.append(traj.openmm_positions(0))
            new_init_box_vecs.append(traj.openmm_boxes(0))
        
        self.init_positions = new_init_positions
        self.init_box_vecs = new_init_box_vecs
        self.init_velocities = None


    