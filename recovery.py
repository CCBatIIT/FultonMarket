"""
Run this script with OUT_OF_MEMORY error occurs and then continue using RUN_FULTONMARKET.py

Provide the path to the replica exchange directory of the simulation you want to recovery ex: /path/to/replica_exchange/SIMNAME_REP
"""
import os, sys
import numpy as np
import netCDF4 as nc
from openmmtools.multistate import MultiStateReporter
from FultonMarket.FultonMarketUtils import truncate_ncdf


# Set output directory
output_dir = sys.argv[1]
save_dir = os.path.join(output_dir, 'saved_variables')
sub_sim_save_dir = os.path.join(save_dir, str(len(os.listdir(save_dir))-1))

# Create reporter
ncdf_fn = os.path.join(output_dir, 'output.ncdf')
reporter = MultiStateReporter(ncdf_fn)
reporter.open()

# Truncate
pos, velos, box_vecs, states, energies, temps = truncate_ncdf(ncdf_fn, 'output.ncdf', sub_sim_save_dir, reporter)

# Save
assert energies.shape[0] > 10, 'this output.ncdf does not have data, please delete and resume simulation.'
np.save(os.path.join(sub_sim_save_dir, 'positions.npy'), pos.data)
np.save(os.path.join(sub_sim_save_dir, 'velocities.npy'), velos.data)
np.save(os.path.join(sub_sim_save_dir, 'box_vectors.npy'), box_vecs.data)
np.save(os.path.join(sub_sim_save_dir, 'states.npy'), states.data)
np.save(os.path.join(sub_sim_save_dir, 'energies.npy'), energies.data)
np.save(os.path.join(sub_sim_save_dir, 'temperatures.npy'), temps.data)



