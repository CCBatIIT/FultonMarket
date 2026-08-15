"""
Imports within this module

Lambda Functions and Variables
geometric_distribution(a, b, c) - create a geometric (logarithmic) distribution from a to b with c values
spring_constant_unit - openmm.unit with units of Joule/(mol*Angstrom^2)
rmsd(a, b) - If a is a single frame and b a trajectory, calculates the rmsd of traj b w.r.t. a
             If a and b are trajectories of the same size, calculates the frame to frame
printf(string) - Shorthand print statement, also prints the current date and time, invoking the flush=True argument

Functions
convert_to_TrackedQuantity(np.array, openmm.unit) - Converts a numpy array to a tracked Quantity array with the provided unit
swap_traj_env(traj1, traj2)
build_thermodynamic_states(self)
restrain_atoms(self)
build_sampler_states(self)
extract_ncdf(ncdf_in, out_dir, reporter) - dump the simulation arrays of an output .ncdf to .npy files
make_interpolated_positions_array(spring_centers1_pdb, spring_centers2_pdb, num_replicates)
make_interpolated_positions_array_from_selections(spring_centers1_pdb, selection_1, spring_centers2_pdb, num_replicates, selection_2=None):
restrain_atoms_by_dsl(thermodynamic_state, topology, atoms_dsl, spring_constant, spring_center)
restrain_atoms_by_index(thermodynamic_state, restrained_atom_indices, spring_constant, spring_center)
restrain_openmm_system_by_dsl(openmm_system, topology, atoms_dsl, spring_constant, spring_center, preselected_centers=True)
"""

import numpy as np
import netCDF4 as nc
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.utils.utils import TrackedQuantity
import mdtraj as md
from openmm import *
from openmm.app import *
import openmm.unit as unit
import math, mpiplus
from datetime import datetime
from copy import deepcopy

geometric_distribution = lambda min_val, max_val, n_vals: [min_val + (max_val - min_val) * (math.exp(float(i) / float(n_vals-1)) - 1.0) / (math.e - 1.0) for i in range(n_vals)]

spring_constant_unit = (unit.joule)/(unit.angstrom*unit.angstrom*unit.mole)

rmsd = lambda a, b: np.sqrt(np.mean(np.sum((b-a)**2, axis=-1), axis=-1))

printf = lambda x: print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + x, flush=True)


def convert_to_TrackedQuantity(arr: np.array, u: openmm.unit):
    return TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=arr, mask=False, fill_value=1e+20), unit=u))


def _interpolate_new_states(prev_temps, insert_inds):

    # Add new states
    new_temps = [temp for temp in prev_temps]
    for displacement, ind in enumerate(insert_inds):
        temp_below = prev_temps[ind-1]
        temp_above = prev_temps[ind]
        print(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + '//' + 'Inserting state at', np.mean((temp_below, temp_above)), flush=True) 
        new_temps.insert(ind + displacement, np.mean((temp_below, temp_above)))
    temperatures = [temp*unit.kelvin for temp in new_temps]
    n_replicates = len(temperatures)
    
    return temperatures, n_replicates


def _interpolate_new_positions(init_positions, init_box_vectors, init_velocities, insert_inds, n_replicates):
    
    # Add pos, box_vecs, velos for new temperatures
    init_positions = np.insert(init_positions, insert_inds, [init_positions[ind-1] for ind in insert_inds], axis=0)
    init_box_vectors = np.insert(init_box_vectors, insert_inds, [init_box_vectors[ind-1] for ind in insert_inds], axis=0)
    if init_velocities is not None:
        init_velocities = np.insert(init_velocities, insert_inds, [init_velocities[ind-1] for ind in insert_inds], axis=0)

    # Convert to quantities    
    init_positions = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_positions, mask=False, fill_value=1e+20), unit=unit.nanometer))
    init_box_vectors = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_box_vectors.reshape(n_replicates, 3, 3), mask=False, fill_value=1e+20), unit=unit.nanometer))

    if init_velocities is not None:
        init_velocities = TrackedQuantity(unit.Quantity(value=np.ma.masked_array(data=init_velocities, mask=False, fill_value=1e+20), unit=(unit.nanometer / unit.picosecond)))

    return init_positions, init_box_vectors, init_velocities


def build_thermodynamic_states(self):

    # Build thermodynamic states
    printf(f'Creating {len(self.temperatures)} Thermodynamic States')
    self.thermodynamic_states = [ThermodynamicState(system=self.system, temperature=T) for T in self.temperatures]
    printf('Done Creating Thermodynamic States')
    printf(f'Assigning {len(self.spring_centers)} Restraints')
    assert len(self.temperatures) == len(self.spring_centers)


def build_sampler_states(n_replicates: int, pos: np.array, box_vec: np.array, velos: np.array=None):

    if velos is not None:
        return [SamplerState(positions=pos[i], box_vectors=box_vec[i], velocities=velos[i]) for i in range(n_replicates)]

    else:
        return [SamplerState(positions=pos[i], box_vectors=box_vec[i]) for i in range(n_replicates)]



def extract_ncdf(ncdf_in, out_dir, reporter):
    """
    Extract the simulation arrays from an output .ncdf into .npy files in out_dir.

    The .ncdf files themselves are deleted by the caller once this returns, so nothing is
    copied or trimmed here. Previously this wrote a 10-iteration stub of the analysis file
    alongside a 1-iteration stub of the checkpoint file with 'last_iteration' forced to 9;
    that pairing is unresumable (openmmtools derives the checkpoint iteration from the
    analysis file's last_iteration, and silently substitutes all-zero coordinates when the
    record is missing), so the stub is no longer produced.
    """
    print(f'Extracting arrays from {ncdf_in}')

    src = nc.Dataset(ncdf_in, 'r')

    for var_name, var in src.variables.items():
        if var_name == 'positions':
            nframes, nstates, natoms, _ = var.shape
            pos_fn = os.path.join(out_dir, 'positions.npy')
            pos = np.memmap(pos_fn, dtype='float32', mode='w+', shape=var.shape)
            print(f'Saving positions to {pos_fn} with shape {var.shape}')
            for frame in range(nframes):
                pos[frame] = var[frame].data
            pos.flush()
        elif var_name == 'box_vectors':
            box_vecs = var[:].copy()
        elif var_name == 'states':
            states = var[:].copy()
        elif var_name == 'energies':
            energies = var[:].copy().astype('float32')
        elif var_name == 'velocities':
            velocities = var[-1].copy().astype('float16')

    src.close()

    # Read temperatures
    temps = np.array([state.temperature._value for state in reporter.read_thermodynamic_states()[0]])

    # Close reporter
    reporter.close()

    return pos, velocities, box_vecs, states, energies, temps
        




        