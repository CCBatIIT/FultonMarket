#Package Imports
from openmm import *
from openmm.app import *
import mdtraj as md
import os, faulthandler
import numpy as np

#Custom Imports
try:
    from .FultonMarketUtils import *
    from .FultonMarket import FultonMarket
except:
    from FultonMarketUtils import *
    from FultonMarket import FultonMarket

#Set some things
np.seterr(divide='ignore', invalid='ignore')
faulthandler.enable()


class FultonMarketPTwFR(FultonMarket):
    """
    Parallel tempering with restraints

    Methods Custom to This Class
        def __init__(self, input_pdb: str, input_system: str, restrained_atoms_dsl: str,
                     input_state: str=None, K=83.68*spring_constant_unit, T_min: float=310, 
                     T_max: float=367.447, n_replicates: int=12)
        
        def _get_restrained_atoms(self)

    Overwrites to Inherited Methods:
        def _set_parameters(self)
        
        def _build_states(self)
        
        def _build_sampler_states(self)
        
        def _build_thermodynamic_states(self)
        
        def _save_sub_simulation(self)
        
        def _load_initial_args(self)

    Methods Inherited from FultonMarket:
        run(self, total_sim_time: float, iter_length: float, dt: float=2.0, sim_length=50,
            init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35, output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/'))
        
        _set_init_positions(self)
        
        _set_init_box_vectors(self)
        
        _configure_experiment_parameters(self, sim_length=50)
        
        _recover_arguments(self)
    """

    def __init__(self, 
                 input_pdb: str, 
                 input_system: str, 
                 restrained_atoms_dsl: str,
                 input_state: str=None,
                 K=83.68*spring_constant_unit,
                 T_min: float=310, 
                 T_max: float=367.447, 
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
        printf('Welcome to FultonMarketPTwR')
        super().__init__(input_pdb=input_pdb,
                         input_system=input_system,
                         input_state=input_state,
                         T_min=T_min,
                         T_max=T_max,
                         n_replicates=n_replicates)
        printf("Finished Initializing Fulton Market PT with Restraints")

        # Set attr
        self.restrained_atoms_dsl = restrained_atoms_dsl
        self.K = K

        # Get restrained atoms
        self._get_restrained_atoms()



    def _get_restrained_atoms(self):
        """
        """

        # Open mdtraj obj
        traj = md.load_pdb(self.input_pdb)
        mdtraj_topology = traj.topology
    
        # Get indices of the atoms to restrain
        self.restrained_atom_indices = mdtraj_topology.select(self.restrained_atoms_dsl)
        if len(self.restrained_atom_indices) == 0:
            raise Exception('No Atoms To Restrain!')

        # Get spring centers
        spring_centers = traj.xyz[0]
        self.spring_centers = np.repeat(spring_centers[np.newaxis, :, :], self.n_replicates, axis=0)
        assert len(self.temperatures) == self.spring_centers.shape[0]
        printf('Restraining All States to the Initial Positions of provided selection string')

    
    def _set_parameters(self):

        # Set parameters for Randolph
        super()._set_parameters()
        self.params['spring_centers'] = self.spring_centers
        self.params['restrained_atom_indices'] = self.restrained_atom_indices

    

    def _build_states(self):

        # Build sampler states
        self._build_sampler_states()
        
        # Build thermodynamic states
        self._build_thermodynamic_states()



    def _build_sampler_states(self):
        
        # Build sampler states
        if self.sim_no == 0 or (hasattr(self, 'init_velocities') and self.init_velocities is None):
            printf('Setting initial positions with the "No Velocities" method')
            self.sampler_states = build_sampler_states(self, self.init_positions, self.init_box_vectors, None)
        else:
            printf('Setting initial positions with the "Velocity" method')
            self.sampler_states = build_sampler_states(self, self.init_positions, self.init_box_vectors, self.init_velocities)


    
    def _build_thermodynamic_states(self):
        build_thermodynamic_states(self)

    
    
    def _save_sub_simulation(self):
        
        # Save temperatures
        self.n_replicates, self.temperatures, self.spring_centers = self.simulation.save_simulation(self.save_dir)


    
    def _load_initial_args(self):
        
        # Load with super
        super()._load_initial_args()
        self.spring_centers = np.load(os.path.join(self.load_dir, 'spring_centers.npy'))
