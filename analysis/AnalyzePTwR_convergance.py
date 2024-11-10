import os, sys
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from FultonMarketAnalysis import FultonMarketAnalysis
from pymbar import MBAR, timeseries


def free_energy_convergence(self, start=1000, size_domain=50):
    
    # Get average energies
    state_energies = np.empty((self.energies.shape[0], self.energies.shape[1]))
    for state in range(self.energies.shape[1]):
        state_energies[:,state] = self.get_state_energies(state)
    average_reduced_potentials = state_energies.mean(axis=1)   

    #Average reduced potentials post equil
    post_equil_average = average_reduced_potentials[self.t0:]
    
    #Full set post equil
    post_equil = self.energies[self.t0:]

    xs = []
    ys = []
    ys_stds = []
    slice_vals = np.arange(start, self.n_frames+size_domain, size_domain, dtype=int)
    for slice_val in slice_vals:
        try:
            print('computing dG for', self.t0, 'to', slice_val)
            
            #Obtain indices of these that are uncorrelated
            indices = np.array(timeseries.subsample_correlated_data(post_equil_average[:slice_val-self.t0], g=self.g))
        
            #obtain the samples that are uncorrelated based on these indices
            uncorrelated_samples = post_equil[:slice_val-self.t0][indices]
            N_k = [uncorrelated_samples.shape[0] for i in range(self.energies.shape[1])]
            mbar = MBAR(uncorrelated_samples.T, N_k, initialize='BAR')
            
            #Obtain dG
            dGs = mbar.compute_free_energy_differences()
    
            xs.append(slice_val)
            ys.append(dGs['Delta_f'][0, -1])
            ys_stds.append(dGs['dDelta_f'][0, -1])
        except:
            print('Could not find solution for',  slice_val, 'to', self.energies.shape[0])


    return np.array(xs), np.array(ys), np.array(ys_stds)


# Initializer analyzer object
centroid = sys.argv[1]
centroid_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/centroids/'
sim_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/PTwR/'
direc, pdbfn = [os.path.join(sim_dir, centroid + '_0'), os.path.join(centroid_dir, centroid + '.pdb')]

analyzer = FultonMarketAnalysis(direc, pdbfn, skip=10)
analyzer.equilibration_method = 'energy'


# Compute free energy convergance
analyzer.equilibration_method = 'energy'
analyzer._determine_equilibration()
xs, ys, y_stds = free_energy_convergence(analyzer, start=analyzer.t0+100, size_domain=100)
np.save(os.path.join(sim_dir, centroid + '_0', 'convergance.npy'), np.array([xs, ys, y_stds]))



