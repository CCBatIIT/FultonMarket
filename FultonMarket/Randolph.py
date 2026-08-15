from openmm import *
from openmm.app import *
from openmmtools import cache
from openmmtools.utils import get_fastest_platform
from openmmtools.utils.utils import TrackedQuantity
from openmmtools import states, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.multistate import ParallelTemperingSampler, ReplicaExchangeSampler, MultiStateReporter
import tempfile
import os, sys, json, shutil
sys.path.append('../MotorRow')
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import netCDF4 as nc
from typing import List
from datetime import datetime
import mdtraj as md
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from FultonMarketUtils import *
from FultonMarketUtils import _interpolate_new_states, _interpolate_new_positions # Not being imported in previous line for some reason
import mpiplus
# from mpi4py import MPI

class Randolph():
    """
    """
    
    def __init__(self, 
                 sampler_states,
                 thermodynamic_states,
                 sim_no: int, 
                 sim_time: unit.Quantity, 
                 temperatures: np.array,
                 output_dir: str, 
                 output_ncdf: str, 
                 checkpoint_ncdf: str,
                 iter_length: unit.Quantity, 
                 dt: unit.Quantity,
                 spring_centers: np.array=None,
                 restrained_atom_indices: np.array=None):
        """
        """
        # Assign attributes
        self.sampler_states = sampler_states
        self.thermodynamic_states = thermodynamic_states
        self.sim_no = sim_no
        self.sim_time = sim_time
        self.output_dir = output_dir
        self.output_ncdf = output_ncdf
        self.checkpoint_ncdf = checkpoint_ncdf
        self.marker_fn = os.path.join(output_dir, '.sim_in_progress.json')
        self.temperatures = temperatures.copy()
        self.n_replicates = len(self.temperatures)
        self.iter_length = iter_length
        self.dt = dt
        self.spring_centers = spring_centers
        if self.spring_centers is not None:
            raise NotImplementedError('Interpolation has been deprecated w/ restraints')
        self.restrained_atom_indices = restrained_atom_indices
        
        # Cycle to start from, overwritten by _adopt_restored_state on a resume
        self.start_cycle = 0

        # Configure simulation parameters
        self._configure_simulation_parameters()

        # Build simulation, resuming from an interrupted sub-simulation if one is on disk
        self._build_simulation(allow_resume=True)


    def main(self, init_overlap_thresh: float, term_overlap_thresh: float):
        """
        """

        # Assign attributes
        self.init_overlap_thresh = init_overlap_thresh
        self.term_overlap_thresh = term_overlap_thresh

        # Continue until self.n_cycles reached
        self.current_cycle = self.start_cycle
        while self.current_cycle <= self.n_cycles:

            # Advance 1 cycle
            self._run_cycle()

            
    @mpiplus.on_single_node(0, broadcast_result=True, sync_nodes=True)
    def save_simulation(self, save_dir):
        """
        Save the important information from a simulation and then remove the .ncdf files to preserve disk space.

        Arrays are written to a temporary directory and renamed into place, so a save dir is
        all-or-nothing: a crash partway through leaves no half-written directory behind. The
        .ncdf files are only removed after the save dir is committed, and the resume marker is
        removed in between, so an interruption at any point leaves a state the next run can
        classify unambiguously.
        """
        # Stage into a temporary directory so the committed save dir is never partial
        save_no_dir = os.path.join(save_dir, str(self.sim_no))
        tmp_dir = os.path.join(save_dir, f'.tmp_{self.sim_no}')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        # Extract arrays from output.ncdf
        pos_memmap, velos, box_vectors, states, energies, temperatures = extract_ncdf(self.output_ncdf, tmp_dir, self.reporter)
        np.save(os.path.join(tmp_dir, 'velocities.npy'), velos.data)
        del velos
        np.save(os.path.join(tmp_dir, 'box_vectors.npy'), box_vectors.data)
        del box_vectors
        np.save(os.path.join(tmp_dir, 'states.npy'), states.data)
        del states
        np.save(os.path.join(tmp_dir, 'energies.npy'), energies.data)
        del energies
        np.save(os.path.join(tmp_dir, 'temperatures.npy'), temperatures)

        if self.spring_centers is not None:
            np.save(os.path.join(tmp_dir, 'spring_centers.npy'), self.spring_centers)

        # Release the positions memmap before renaming the directory out from under it
        del pos_memmap

        # Close reporter object
        try:
            self.reporter.close()
        except:
            pass

        # Commit the save dir, then retire the .ncdf files it was built from
        if os.path.exists(save_no_dir):
            shutil.rmtree(save_no_dir)
        os.rename(tmp_dir, save_no_dir)
        self._do_remove_marker()
        self._do_remove_ncdf()

        if self.spring_centers is not None:
            return len(temperatures), [t*unit.kelvin for t in temperatures], self.spring_centers
        else:
            return len(temperatures), [t*unit.kelvin for t in temperatures]
        

    
    def _configure_simulation_parameters(self):
        """
        Configure simulation times to meet aggregate simulation time. 
        """            

        # Read number replicates if different than argument
        self.n_replicates = len(self.temperatures)
        
        # Configure times/steps
        sim_time_per_rep = self.sim_time / self.n_replicates
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated simulation per replicate to be', np.round(sim_time_per_rep, 6), 'nanoseconds', flush=True)
        
        steps_per_rep = np.ceil(sim_time_per_rep * 1e6 / self.dt)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per replicate to be', np.round(steps_per_rep,0), 'steps', flush=True)        
        
        self.n_steps_per_iter = self.iter_length * 1e6 / self.dt
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated steps per iteration to be', np.round(self.n_steps_per_iter, 0), 'steps', flush=True) 
        
        self.n_iters = np.ceil(steps_per_rep / self.n_steps_per_iter)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iterations to be', self.n_iters, 'iterations', flush=True) 
        
        self.n_cycles = np.ceil(self.n_iters / 5)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of cycles to be', self.n_cycles, 'cycles', flush=True) 
        
        self.n_iters_per_cycle = np.ceil(self.n_iters / self.n_cycles)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated number of iters per cycle to be', self.n_iters_per_cycle, 'iterations', flush=True) 

#        self.checkpoint_interval = int(0.01 / self.iter_length)
        self.checkpoint_interval = 1
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated checkpoint interval to be', self.checkpoint_interval, 'iterations', flush=True) 


        # Configure replicates            
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Calculated temperature of', self.n_replicates,
                                      'replicates to be', [np.round(t._value,1) for t in self.temperatures], flush=True)


    # NOTE: the _do_* methods below are plain (undecorated) so they can be called from inside
    # save_simulation, which already runs on rank 0 only. Nesting an on_single_node call with
    # sync_nodes inside a rank-0-only block would deadlock, since the other ranks never reach
    # the barrier. The decorated wrappers are for the all-ranks call sites.

    def _do_remove_ncdf(self):
        for ncdf in (self.output_ncdf, self.checkpoint_ncdf):
            if os.path.exists(ncdf):
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Removing', ncdf, flush=True)
                os.remove(ncdf)


    def _do_write_marker(self):
        """
        Record which sub-simulation the .ncdf files on disk belong to, and the run parameters
        they were produced with. Presence of this file is what makes an interrupted
        sub-simulation resumable; save_simulation removes it once the save dir is committed.
        """
        marker = dict(sim_no=int(self.sim_no),
                      n_replicates=int(self.n_replicates),
                      iter_length=float(self.iter_length),
                      dt=float(self.dt),
                      sim_time=float(self.sim_time))
        with open(self.marker_fn, 'w') as f:
            json.dump(marker, f)


    def _do_remove_marker(self):
        if os.path.exists(self.marker_fn):
            os.remove(self.marker_fn)


    @mpiplus.on_single_node(0, sync_nodes=True)
    def _remove_ncdf(self):
        self._do_remove_ncdf()


    @mpiplus.on_single_node(0, sync_nodes=True)
    def _write_marker(self):
        self._do_write_marker()


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True)
    def _can_resume(self):
        """
        Decide whether the .ncdf files on disk are an interrupted run of THIS sub-simulation.

        Deliberately marker-driven rather than inspecting the .ncdf: openmmtools performs no
        cross-file validation, and reading a checkpoint record that is absent yields all-zero
        coordinates instead of an error. Anything short of an exact match starts fresh.
        """
        # Marker must exist and belong to this sub-simulation
        if not os.path.exists(self.marker_fn):
            return False
        try:
            with open(self.marker_fn, 'r') as f:
                marker = json.load(f)
        except (ValueError, OSError) as e:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'Unreadable resume marker ({e}), starting sub-simulation fresh', flush=True)
            return False

        if marker.get('sim_no') != self.sim_no:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f"Resume marker is for sim_no {marker.get('sim_no')}, but this is sim_no {self.sim_no}. Discarding stale .ncdf files.", flush=True)
            return False

        # Both .ncdf files must be present and non-empty (openmmtools storage_exists requires both)
        for ncdf in (self.output_ncdf, self.checkpoint_ncdf):
            if not os.path.exists(ncdf) or os.path.getsize(ncdf) == 0:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'{ncdf} missing or empty, starting sub-simulation fresh', flush=True)
                return False

        # Run parameters must match: from_storage restores mcmc_moves and number_of_iterations
        # from the file, so a changed timestep/iteration length would be silently ignored
        for key, current in (('iter_length', self.iter_length), ('dt', self.dt), ('sim_time', self.sim_time)):
            if marker.get(key) != float(current):
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'Run parameter {key} changed ({marker.get(key)} -> {current}) since the interrupted run. Starting sub-simulation fresh.', flush=True)
                return False

        return True


    def _build_simulation(self, allow_resume: bool=False):
        """
        Build the sampler, either fresh or resumed from an interrupted sub-simulation.

        allow_resume is only True for the initial build. The rebuild that follows state
        interpolation must always be fresh, since the number of thermodynamic states has
        just changed.
        """
        if allow_resume and self._can_resume():
            try:
                self._resume_simulation()
                self._write_marker()
                return
            except Exception as e:
                print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'FAILED to resume from {self.output_ncdf} ({type(e).__name__}: {e}). Starting sub-simulation fresh.', flush=True)
                # Release any handles the partial resume opened before _create_simulation
                # unlinks the files underneath them
                try:
                    self.reporter.close()
                except:
                    pass

        self._create_simulation()
        self._write_marker()


    def _create_simulation(self):
        """
        Create a new sampler, discarding any .ncdf files already in the output directory.
        """
        # Set up integrator
        move = mcmc.LangevinDynamicsMove(timestep=self.dt * unit.femtosecond, collision_rate=1.0 / unit.picosecond, n_steps=self.n_steps_per_iter, reassign_velocities=False)

        # Set up simulation
        if self.spring_centers is not None:
            self.simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        else:
            self.simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=self.n_iters)
        self.simulation._global_citation_silence = True

        # Remove existing .ncdf files
        self._remove_ncdf()

        # Setup reporter
        atom_inds = tuple([i for i in range(self.thermodynamic_states[0].system.getNumParticles())])
        self.reporter = MultiStateReporter(self.output_ncdf, checkpoint_interval=self.checkpoint_interval, analysis_particle_indices=atom_inds)

        # Create simulation obj
        if self.spring_centers is not None:
            self.simulation.create(thermodynamic_states=self.thermodynamic_states, sampler_states=self.sampler_states, storage=self.reporter)
        else:
            self.simulation.create(thermodynamic_state=self.thermodynamic_states[0], sampler_states=self.sampler_states,
                                   storage=self.reporter, temperatures=self.temperatures, n_temperatures=self.n_replicates)


    def _resume_simulation(self):
        """
        Restore a sampler from the .ncdf files of an interrupted sub-simulation.
        """
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + f'Resuming sub-simulation {self.sim_no} from {self.output_ncdf}', flush=True)

        atom_inds = tuple([i for i in range(self.thermodynamic_states[0].system.getNumParticles())])
        self.reporter = MultiStateReporter(self.output_ncdf, checkpoint_interval=self.checkpoint_interval, analysis_particle_indices=atom_inds)

        sampler_cls = ReplicaExchangeSampler if self.spring_centers is not None else ParallelTemperingSampler

        # from_storage prints the citation block from inside _instantiate_sampler_from_reporter,
        # before it hands back an instance, so the flag has to be set on the class rather than
        # on self.simulation the way _create_simulation does it.
        sampler_cls._global_citation_silence = True

        self.simulation = sampler_cls.from_storage(self.reporter)
        self.simulation._global_citation_silence = True

        self._adopt_restored_state()


    def _adopt_restored_state(self):
        """
        Take the restored file as authoritative for the state schedule and iteration count.

        State interpolation can add temperatures partway through a sub-simulation, so the
        .ncdf may hold more states than the save dir the caller loaded its arguments from.
        """
        # Sanity check the restored coordinates. openmmtools substitutes all-zero positions
        # when a checkpoint record is missing rather than raising, which would otherwise only
        # surface much later as a NaN energy.
        sampler_states = getattr(self.simulation, 'sampler_states', None)
        if sampler_states is None:
            sampler_states = self.simulation._sampler_states
        for i, sampler_state in enumerate(sampler_states):
            if not np.any(sampler_state.positions._value):
                raise ValueError(f'Restored positions for replicate {i} are all zero, indicating a missing checkpoint record. Refusing to resume.')

        # Adopt the schedule recorded in the file. Read via _read_temps_from_reporter rather
        # than touching self.reporter directly: from_storage reopens the reporter on rank 0
        # only, so a bare read would fail on every other rank.
        self.temperatures = [t*unit.kelvin for t in self._read_temps_from_reporter()]
        self.n_replicates = len(self.temperatures)

        # Adopt the iteration count the file was created with, then redo the cycle arithmetic
        self.n_iters = self.simulation.number_of_iterations
        self.n_cycles = np.ceil(self.n_iters / 5)
        self.n_iters_per_cycle = np.ceil(self.n_iters / self.n_cycles)

        # Skip the cycles already on disk
        restored_iteration = self.simulation._iteration
        self.start_cycle = int(restored_iteration // self.n_iters_per_cycle)

        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Restored', self.n_replicates,
              'replicates at iteration', restored_iteration, 'of', self.n_iters,
              '- resuming at cycle', self.start_cycle, 'of', self.n_cycles, flush=True)
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'NOTE: random number generator state is not persisted by openmmtools, so this run is not a bitwise continuation of the interrupted one.', flush=True)

    
    def _run_cycle(self):
        """
        Run one cycle
        """

        comm = mpiplus.get_mpicomm()
        
        # Take steps
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'CYCLE', self.current_cycle, 'advancing', self.n_iters_per_cycle, 'iterations', flush=True) 
        if self.simulation.is_completed:
            self.simulation.extend(self.n_iters_per_cycle)
        else:
            self.simulation.run(self.n_iters_per_cycle)

        # Eval acceptance rates
        if self.sim_no == 0:
            insert_inds = self._eval_acc_rates(self.init_overlap_thresh)
        else:
            insert_inds = self._eval_acc_rates(self.term_overlap_thresh)

        # Interpolate, if necessary
        if len(insert_inds) > 0:
            self._interpolate_states(insert_inds)
            self.reporter.close()
            self.current_cycle = 0
            self._configure_simulation_parameters()
            self._build_simulation()
        else:
            self.current_cycle += 1


    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True) 
    def _eval_acc_rates(self, acceptance_rate_thresh: float=0.40):
        
        # Get temperatures
        temperatures = [float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]]
        
        # Get mixing statistics
        accepted, proposed = self.reporter.read_mixing_statistics()
        acc_rates = np.mean(accepted[1:] / proposed[1:], axis=0)
        acc_rates = np.nan_to_num(acc_rates) # Adjust for cases with 0 proposed swaps
    
        # Iterate through mixing statistics to flag acceptance rates that are too low
        insert_inds = [] # List of indices to apply new state. Ex: (a "1" means a new state between "0" and the previous "1" indiced state)
        for state in range(len(acc_rates)-1):
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Mixing between', np.round(temperatures[state], 2), 'and', np.round(temperatures[state+1], 2), ':', acc_rates[state, state+1], flush=True) 
            rate = acc_rates[state, state+1]
            if rate < acceptance_rate_thresh:
                insert_inds.append(state+1)
        
        return np.array(insert_inds)

    
    @mpiplus.on_single_node(rank=0, broadcast_result=True, sync_nodes=True) 
    def _read_temps_from_reporter(self): 
        return np.array([float(s.temperature._value) for s in self.reporter.read_thermodynamic_states()[0]])

        
    def _interpolate_states(self, insert_inds: np.array):
        
        # Determine new states
        prev_temps = self._read_temps_from_reporter()
        self.temperatures, self.n_replicates = _interpolate_new_states(prev_temps, insert_inds)
        init_positions, init_box_vectors, init_velocities = self._load_inits()
        init_positions, init_box_vectors, init_velocities = _interpolate_new_positions(init_positions, init_box_vectors, init_velocities, insert_inds, self.n_replicates)
        
        # Update Sampler States
        self.sampler_states = build_sampler_states(self.n_replicates, init_positions, init_box_vectors, init_velocities)

        
        # Add new restraints, if necessary, MAY BE DEPRECATED w/ MPI implementation
        # if self.spring_centers is not None:
        #     prev_spring_centers = self.spring_centers
        #     new_spring_centers = self.spring_centers
        #     for displacement, ind in enumerate(insert_inds):
        #         center_below = prev_spring_centers[ind - 1]
        #         center_above = prev_spring_centers[ind]
        #         print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state with new Spring Center', flush=True)
        #         new_center = 0.5*(center_above + center_below)
        #         new_spring_centers = np.insert(new_spring_centers, ind + displacement, new_center, axis=0)
        #     self.spring_centers = new_spring_centers
        #     assert self.spring_centers.shape[0] == len(self.temperatures)

        #     # Update Thermodynamic States
        #     self.system = self.thermodynamic_states[0].system
        #     self.system.removeForce(6)# Remove previous CustomExternalForce
        #     build_thermodynamic_states(self)


    def _load_inits(self):

        init_positions = np.array([self.sampler_states[i].positions._value.copy() for i in range(len(self.sampler_states))])
        init_box_vectors = np.array([self.sampler_states[i].box_vectors._value.copy() for i in range(len(self.sampler_states))])
        if self.sim_no > 0 and self.sampler_states[0].velocities is not None:
            init_velocities = np.array([self.sampler_states[i].velocities._value.copy() for i in range(len(self.sampler_states))])
        else:
            init_velocities = None

        return init_positions, init_box_vectors, init_velocities
            

