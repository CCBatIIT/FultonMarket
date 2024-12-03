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

#Custom Imports (Try Relative First)
try:
    from .FultonMarketUtils import *
    from .FultonMarketPTwFR import FultonMarketPTwFR
    from .analysis.FultonMarketAnalysis import FultonMarketAnalysis
except:
    from FultonMarketUtils import *
    from FultonMarketPTwFR import FultonMarketPTwFR
    from analysis.FultonMarketAnalysis import FultonMarketAnalysis


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
    
        _get_restrained_atoms(self)
        
        _set_init_positions(self)
        
        _set_init_box_vectors(self)

        _load_initial_args(self)
        
    Methods Inherited from FultonMarketPTwFR:    
    
        _set_parameters(self)
        
        _build_states(self)
        
        _build_sampler_states(self)
        
        _build_thermodynamic_states(self)
        
        _save_sub_simulation(self)
        
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
                 init_positions_dcd: str, 
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
        printf(f"Assigning Initial Positions from Unilaterl Trailblazing DCD {self.init_positions_dcd}")
        assert os.path.exists(self.init_positions_dcd), self.init_positions_dcd

        # Load traj objs
        init_traj = md.load(self.init_positions_dcd, top=self.input_pdb[0])
        printf(f"Read {init_traj.n_frames} number of states from trailblazing")

        # Get pos, box_vectors
        self.init_positions = [init_traj.openmm_positions(i) for i in range(init_traj.n_frames)]
        self.init_box_vectors = [init_traj.openmm_boxes(i) for i in range(init_traj.n_frames)]
        #self.init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_traj.xyz, mask=False, fill_value=1e+20), unit=unit.nanometer))
        #self.init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_traj.unitcell_vectors, mask=False, fill_value=1e+20), unit=unit.nanometer))
    
    
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



    def _load_initial_args(self):

        # Get last directory
        load_no = self.sim_no - 1
        self.load_dir = os.path.join(self.save_dir, str(load_no))
        
        # Load temps
        self.temperatures = np.load(os.path.join(self.load_dir, 'temperatures.npy'))
        self.temperatures = [t*unit.kelvin for t in self.temperatures]

        # Load spring centers
        self.spring_centers = np.load(os.path.join(self.load_dir, 'spring_centers.npy'))

        # Get pos, box_vec by resampling
        self._resample_init_positions()


    
    def _resample_init_positions(self):
        """
        """
        printf(f"Begin Resampling {len(self.temperatures)} positions...")
        #Load an analyzer
        input_dir = os.path.abspath(os.path.join(self.save_dir, '..'))
        #The analyzer will handle the loading of energies and any backfilling
        analyzer = FultonMarketAnalysis(input_dir, self.input_pdb[0], scheduling='Spring Centers')
        #Set the new intial positions and box vecs by resampling
        new_init_positions = []
        new_init_box_vecs = []
        for i in range(len(self.temperatures)):
            analyzer.importance_resampling(n_samples=1, equilibration_method='None', specify_state=i)
            #sets analyzer.resampled_inds and analyzer.weights
            temp_pdb_fn, temp_dcd_fn = os.path.join(self.output_dir, 'temp.pdb'), os.path.join(self.output_dir, 'temp.dcd')
            traj = analyzer.write_resampled_traj(temp_pdb_fn, temp_dcd_fn, return_traj=True)
            #clean up from that line
            os.remove(temp_pdb_fn)
            os.remove(temp_dcd_fn)
            #Add positions and box vectors to the list
            new_init_positions.append(traj.xyz[0])
            new_init_box_vecs.append(traj.unitcell_vectors[0])
        
        self.init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=new_init_positions, mask=False, fill_value=1e+20), unit=unit.nanometer))
        self.init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=new_init_box_vecs, mask=False, fill_value=1e+20), unit=unit.nanometer))
        self.init_velocities = None
        printf(f"Successfully Resampled {len(self.temperatures)} positions.")


    
