from openmm import *
from openmm.app import *
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.multistate import ParallelTemperingSampler, MultiStateReporter
from openmmtools.utils.utils import TrackedQuantity
from openmm.unit.quantity import Quantity
from openmm.vec3 import Vec3
import tempfile
import os, sys, math
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import netCDF4 as nc
from typing import List
from datetime import datetime
import mdtraj as md
from FultonMarketUtils import *
from FultonMarket import FultonMarket
from FultonMarketPTwFR import FultonMarketPTwFR
from Randolph import Randolph
import faulthandler
from copy import deepcopy
faulthandler.enable()


class FultonMarketUS(FultonMarketPTwFR):
    """
    Parallel tempering with restraints
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
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Welcome to FultonMarketUS.', flush=True)

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


    