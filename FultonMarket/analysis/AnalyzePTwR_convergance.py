import os, sys
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from FultonMarketAnalysis import FultonMarketAnalysis
from pymbar import MBAR, timeseries


def free_energy_convergence(self, size_domain=50):
    

    # Get average energies
    state_energies = np.empty((self.energies.shape[0], self.energies.shape[1]))
    for state in range(self.energies.shape[1]):
        state_energies[:,state] = self.get_state_energies(state)
    average_reduced_potentials = state_energies.mean(axis=1)   

    # Get equilibration
    #detect equil based on average reduced potential
    t0, g, Neff_max = timeseries.detect_equilibration(average_reduced_potentials)
    print('Equilibration detected at', t0)
    #Average reduced potentials post equil
    post_equil_average = average_reduced_potentials[t0:]
    #Full set post equil
    post_equil = self.energies[t0:]

    
    xs = []
    ys = []
    ys_stds = []
    slice_vals = np.arange(t0+size_domain, self.energies.shape[0]+size_domain, size_domain, dtype=int)
    for slice_val in slice_vals:
        try:
            print('computing dG for', t0, 'to', slice_val, flush=True)
            
            #Obtain indices of these that are uncorrelated
            indices = np.array(timeseries.subsample_correlated_data(post_equil_average[:slice_val-t0], g=g))
        
            #obtain the samples that are uncorrelated based on these indices
            uncorrelated_samples = post_equil[:slice_val-t0][indices]
            N_k = [uncorrelated_samples.shape[0] for i in range(self.energies.shape[1])]
            mbar = MBAR(uncorrelated_samples.T, N_k, initialize='BAR')
            
            #Obtain dG
            dGs = mbar.compute_free_energy_differences()
    
            xs.append(slice_val)
            ys.append(dGs['Delta_f'][0, -1])
            ys_stds.append(dGs['dDelta_f'][0, -1])
        except:
            print('Could not find solution for',  t0, 'to', slice_val, flush=True)


    return np.array(xs), np.array(ys), np.array(ys_stds)


# Initializer analyzer object
centroid = sys.argv[1]
centroid_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/centroids/'
sim_dir = '/expanse/lustre/projects/iit122/dcooper/CB2/PTwR/'
direc, pdbfn = [os.path.join(sim_dir, centroid + '_0'), os.path.join(centroid_dir, centroid + '.pdb')]

analyzer = FultonMarketAnalysis(direc, pdbfn, skip=10)
analyzer.equilibration_method = 'energy'


# Compute free energy convergance
xss = []
yss = []
y_stdss = []
xs, ys, y_stds = free_energy_convergence(analyzer, size_domain=100)
xss.append(xs)
yss.append(ys)
y_stdss.append(y_stds)
plt.clf()
_ = plt.errorbar(xs/10, ys, yerr=y_stds, ecolor='gray', capsize=2.5)
plt.title(centroid)
plt.ylabel('dG between endstates (kT)')
plt.xlabel('Aggregate Simulation Time (ns)')
plt.savefig(os.path.join(sim_dir, f'{centroid}_dG_convergance.png')) 



