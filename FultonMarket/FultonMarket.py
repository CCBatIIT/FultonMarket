#Package Imports
from openmm import *
from openmm.app import *
from openmmtools.states import SamplerState, ThermodynamicState
import numpy as np
import netCDF4 as nc
import os, sys, faulthandler

#Custom Imports
from .FultonMarketUtils import *
from .Randolph import Randolph

#Set some things
np.seterr(divide='ignore', invalid='ignore')
sys.path.append('../MotorRow')
faulthandler.enable()


class FultonMarket():
    """
    Unrestrained Replica Exchange

    Methods:
         __init__(self, input_pdb: str, input_system: str, input_state: str=None, T_min: float=300, T_max: float=367.447, n_replicates: int=12)
    
        run(self, total_sim_time: float, iter_length: float, dt: float=2.0, sim_length=50,
            init_overlap_thresh: float=0.5, term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/'))
    
        _set_init_positions(self)
    
        _set_init_box_vectors(self)
    
        _set_parameters(self)
    
        _build_states(self)
    
        _build_sampler_states(self)
    
        _build_thermodynamic_states(self)
    
        _save_sub_simulation(self)
    
        _load_initial_args(self)
    
        _configure_experiment_parameters(self, sim_length=50)
    
        _recover_arguments(self)
    """

    def __init__(self, 
                 input_pdb: str, 
                 input_system: str, 
                 input_state: str,
                 T_min: float=300, 
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
        printf('Welcome to FultonMarket.')


        # Set attr
        self.temperatures = [temp*unit.kelvin for temp in geometric_distribution(T_min, T_max, n_replicates)]
        self.n_replicates = n_replicates

        # Unpack .pdb
        self.input_pdb = input_pdb
        self.pdb = PDBFile(input_pdb)
        self._set_init_positions()
        printf(f'Found input_pdb: {input_pdb}')

        # Unpack .xml
        self.system = XmlSerializer.deserialize(open(input_system, 'r').read())
        self._set_init_box_vectors()
        printf(f'Found input_system: {input_system}')

        # Build state
        if input_state != None:
            integrator = LangevinIntegrator(300, 0.01, 2)
            sim = Simulation(self.pdb.topology, self.system, integrator)
            sim.loadState(input_state)
            self.context = sim.context
            printf(f'Found input_state: {input_state}')



    def _set_init_positions(self):

        # Repeat pdb positions
        self.init_positions = self.pdb.getPositions(asNumpy=True)
        self.init_positions = [self.init_positions for i in range(self.n_replicates)]

    

    def _set_init_box_vectors(self):

        # Repeat system box vectors
        self.init_box_vectors = self.system.getDefaultPeriodicBoxVectors()
        self.init_box_vectors = [self.init_box_vectors for i in range(self.n_replicates)]
        
        
    
    def run(self, 
            total_sim_time: float, 
            iter_length: float, 
            dt: float=2.0, 
            sim_length=50,
            init_overlap_thresh: float=0.5, 
            term_overlap_thresh: float=0.35,
            output_dir: str=os.path.join(os.getcwd(), 'FultonMarket_output/')):
        
        """
        Run parallel tempering replica exchange.

        Parameters:
        -----------
            total_sim_time (float):
                Aggregate simulation time from all replicates in nanoseconds.

            iter_length (float):
                Specify the amount of time between swapping replicates in nanoseconds. 

            dt (float):
                Timestep for simulation. Default is 2.0 femtoseconds.

            T_min (float):
                Minimum temperature in Kelvin. This state will serve as the reference state. Default is 300 K.

            T_max (float):
                Maximum temperature in Kelvin. Default is 360 K.

            n_replicates (int):
                Number of replicates, meaning number of states between T_min and T_max. States are automatically built at with a geometeric distribution towards T_min. Default is 12.

            init_overlap_thresh (float):
                Acceptance rate threshold during first 50 ns simulation to cause restart. Default is 0.50. 

            term_overlap_thresh (float):
                Terminal acceptance rate. If the minimum acceptance rate every falls below this threshold simulation with restart. Default is 0.35.

            output_dir (str):
                String path to output directory to store files. Default is 'FultonMarket_output' in the current working directory.

        """

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

            # Delete output.ncdf files if not last simulation 
            if not self.sim_no+1 == self.total_n_sims:
                os.remove(self.output_ncdf)
                os.remove(self.checkpoint_ncdf)

            # Update counter
            self.sim_no += 1



    def _set_parameters(self):

        # Set parameters for Randolph
        self.params = dict(sampler_states=self.sampler_states,
                           thermodynamic_states=self.thermodynamic_states,
                           sim_no=self.sim_no,
                           sim_time=self.sim_time,
                           temperatures=self.temperatures,
                           output_dir=self.output_dir,
                           output_ncdf=self.output_ncdf,
                           checkpoint_ncdf=self.checkpoint_ncdf,
                           iter_length=self.iter_length,
                           dt=self.dt)

    

    def _build_states(self):
        
        # Build sampler and thermodyanic states individually
        self._build_sampler_states()
        self._build_thermodynamic_states()

    

    def _build_sampler_states(self):
        
        # Build sampler states
        if self.sim_no == 0:
            printf('Setting initial positions with the "Context" method')
            self.sampler_states = [SamplerState(positions=self.init_positions, box_vectors=self.init_box_vectors).from_context(self.context) for i in range(self.n_replicates)]
        else:
            printf('Setting initial positions with the "Velocity" method')
            self.sampler_states = build_sampler_states(self, self.init_positions, self.init_box_vectors, self.init_velocities)



    def _build_thermodynamic_states(self):
        
        # Build thermodynamic states
        if not hasattr(self, 'thermodynamic_states'):
            self.thermodynamic_states = [ThermodynamicState(system=self.system, temperature=self.temperatures[0], pressure=1.0*unit.bar)] 



    def _save_sub_simulation(self):
        
        # Save temperatures
        self.temperatures = self.simulation.save_simulation(self.save_dir)


    
    def _load_initial_args(self):
        
        # Get last directory
        load_no = self.sim_no - 1
        self.load_dir = os.path.join(self.save_dir, str(load_no))
        
        # Load args (not in correct shapes
        self.temperatures = np.load(os.path.join(self.load_dir, 'temperatures.npy'))
        self.temperatures = [t*unit.kelvin for t in self.temperatures]


        # Load from .npy files
        try:
            init_positions = np.load(os.path.join(self.load_dir, 'positions.npy'))[-1] 
            init_box_vectors = np.load(os.path.join(self.load_dir, 'box_vectors.npy'))[-1] 
            init_velocities = np.load(os.path.join(self.load_dir, 'velocities.npy')) 
            state_inds = np.load(os.path.join(self.load_dir, 'states.npy'))[-1]
        except:
            try:
                init_positions = np.load(os.path.join(self.load_dir, 'positions.npy'))[-1]
                init_box_vectors = np.load(os.path.join(self.load_dir, 'box_vectors.npy'))[-1]
                init_velocities = None
                state_inds = np.load(os.path.join(self.load_dir, 'states.npy'))[-1]
            except:
                init_velocities, init_positions, init_box_vectors, state_inds = self._recover_arguments()
        
        # Reshape 
        reshaped_init_positions = np.empty((init_positions.shape))
        reshaped_init_box_vectors = np.empty((init_box_vectors.shape))
        for state in range(len(self.temperatures)):
            rep_ind = np.where(state_inds == state)[0]
            reshaped_init_box_vectors[state] = init_box_vectors[rep_ind] 
            reshaped_init_positions[state] = init_positions[rep_ind] 
            
        if init_velocities is not None:
            reshaped_init_velocities = np.empty((init_velocities.shape))
            for state in range(len(self.temperatures)):
                rep_ind = np.where(state_inds == state)[0]
                reshaped_init_velocities[state] = init_velocities[rep_ind] 
                
        # Convert to quantities    
        self.init_positions = convert_to_TrackedQuantity(reshaped_init_positions, unit.nanometer)
        self.init_box_vectors = convert_to_TrackedQuantity(reshaped_init_box_vectors, unit.nanometer)
        if init_velocities is not None:
            self.init_velocities = convert_to_TrackedQuantity(reshaped_init_velocities, (unit.nanometer / unit.picosecond))
        else:
            self.init_velocities = None


    
    def _configure_experiment_parameters(self, sim_length=50):
        # Assert that no empty save directories have been made
        assert all([len(os.listdir(os.path.join(self.save_dir, dir))) >= 5 for dir in os.listdir(self.save_dir)]), "You may have an empty save directory, please remove empty or incomplete save directories before continuing :)"
        
        # Configure experiment parameters
        self.sim_no = len(os.listdir(self.save_dir))
        printf(f'Found n_sims_completed to be {self.sim_no}')
        self.sim_time = sim_length # ns
        self.total_n_sims = np.ceil(self.total_sim_time / self.sim_time)
        printf(f'Calculated total_n_sims to be {self.total_n_sims}')


    
    def _recover_arguments(self):
        ncfile = nc.Dataset(self.output_ncdf, 'r')
        
        # Read
        velocities = ncfile.variables['velocities'][-1].data
        positions = ncfile.variables['positions'][-1].data
        box_vectors = ncfile.variables['box_vectors'][-1].data
        state_inds = ncfile.variables['states'][-1].data
        
        ncfile.close()
        
        return velocities, positions, box_vectors, state_inds