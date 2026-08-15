"""
DEPRECATED. This script is no longer needed and no longer works.

It used to salvage a partially-written output.ncdf into the previous sub-simulation's save
directory after an OUT_OF_MEMORY crash, so that RUN_FULTONMARKET.py could be resumed by hand.

FultonMarket now does this automatically. An interrupted sub-simulation is resumed in place
from its own output.ncdf via ParallelTemperingSampler.from_storage(), and runs on to its full
configured length, so the saved_variables/ layout is identical to an uncrashed run. Just
re-run the same RUN_FULTONMARKET.py command; see Randolph._can_resume / _resume_simulation.

This file can be deleted.
"""
import sys

sys.exit(__doc__)
