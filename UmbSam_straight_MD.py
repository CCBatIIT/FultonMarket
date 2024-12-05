##### Run Straight MD of each centroid

#USAGE python US_straight_MD.py input_dir centroid_A_name centroid_B_name, save_dir, trailblazing_dcd_fn, replicate_index

import os, sys
from openmm import *
import numpy as np
from openmm.app import *
import openmm.unit as unit
from datetime import datetime
import mdtraj as md

from FultonMarket.FultonMarketUtils import *


class Straight_MD4US():
    """
    
    """
    
    def __init__(self, input_dir, centroid_A_name, centroid_B_name, save_dir, trailblazing_dcd_fn,
                 replicate_index:int, temp=310*unit.kelvin, pressure=1*unit.bar):
        """
        """

        # Ensure User inputs Exist and Assign them
        assert os.path.exists(input_dir)
        printf(f'Found input directory: {input_dir}')
        centroid_A_pdb = os.path.join(input_dir, centroid_A_name + '.pdb')
        centroid_A_xml = os.path.join(input_dir, centroid_A_name + '_sys.xml')
        assert os.path.exists(centroid_A_pdb) and os.path.exists(centroid_A_xml)
        printf(f'Found Centroid A: {centroid_A_name}')
        centroid_B_pdb = os.path.join(input_dir, centroid_B_name + '.pdb')
        centroid_B_xml = os.path.join(input_dir, centroid_B_name + '_sys.xml')
        assert os.path.exists(centroid_B_pdb) and os.path.exists(centroid_B_xml)
        printf(f'Found Centroid B: {centroid_B_name}')
        self.pdb_fns = {"A": centroid_A_pdb, "B": centroid_B_pdb}
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        assert os.path.exists(save_dir)
        printf(f'Found output directory: {save_dir}')
        self.save_dir = save_dir
        
        # Open input 
        self.sim_obs = get_some_objects(centroid_A_pdb, centroid_A_xml)
        #Sim obs has shape (PDBFile, openmm.System, openmm.Topology, mdtraj.Topology)

        #Output
        self.sim_name = f"{centroid_A_name}_{centroid_B_name}_{replicate_index}"
        self.stdout_fn = os.path.join(save_dir, f'{self.sim_name}.stdout')
        self.dcd_fn = os.path.join(save_dir, f'{self.sim_name}.dcd')
        self.state_fn = os.path.join(save_dir, f'{self.sim_name}_state.xml')
        printf(f'Saving stdout to {self.stdout_fn}, \n dcd to {self.dcd_fn}')
        
        # Simulation physical parameters
        self.temp = temp
        printf(f'Found simulation temperature: {temp}')
        self.pressure = pressure
        printf(f'Found simulation pressure: {self.pressure}')

        #Init Positions should be assigned from the trailblazing DCD
        trail_traj = md.load(trailblazing_dcd_fn, top=self.sim_obs[3])
        assert replicate_index < trail_traj.n_frames
        
        self.replicate_index = replicate_index
        self.n_from_trailblazing = trail_traj.n_frames
        self.init_positions = trail_traj.openmm_positions(replicate_index)
        self.init_boxes = trail_traj.openmm_boxes(replicate_index)
        print(self.init_boxes)


    def run(self, ts=2*unit.femtosecond, sim_time=50*unit.nanosecond, time_btw_frames=5*unit.picosecond):

        # Simulation length parameters
        n_steps_per_frame = round(time_btw_frames / ts)
        nstdout = n_steps_per_frame
        n_frames_total = round(sim_time / time_btw_frames)
        n_steps_total = round(sim_time / ts)

        printf(f'Found timestep of: {ts}')
        printf(f'Found time between frames: {time_btw_frames}')
        printf(f'Found steps per frame: {n_steps_per_frame}')
        printf(f'Found nstdout of: {nstdout}')
        printf(f'Found ndcd of: {time_btw_frames}')
        printf(f'Found total no. of frames: {n_frames_total}')
        
        spring_center = self.spring_centers[self.replicate_index]
        
        # Reporter Parameters
        SDR_params = dict(file=self.stdout_fn, reportInterval=nstdout, step=True, time=True,
                          potentialEnergy=True, temperature=True, progress=False,
                          remainingTime=False, speed=True, volume=True,
                          totalSteps=n_frames_total, separator=' : ')
        DCDR_params = dict(file=self.dcd_fn, reportInterval=n_steps_per_frame, enforcePeriodicBox=True)

        printf(f'Beginning simulation for replicate index {self.replicate_index}...')

        # Deep copy the init system ("A"), without the restraints or barostat, then add them
        system = copy.deepcopy(self.sim_obs[1])
        restrain_openmm_system_by_dsl(system, self.sim_obs[3],
                                      self.restraint_selection_string,
                                      self.spring_constant, spring_center)
        system.addForce(MonteCarloBarostat(self.pressure, self.temp, 100))

        # Define the integrator and simulation etc.
        integrator = LangevinIntegrator(self.temp, 1/unit.picosecond, ts)
        simulation = Simulation(self.sim_obs[2], system, integrator)
        for param_set in [SDR_params, DCDR_params]:
            param_set['append'] = False
        simulation.context.setPositions(self.init_positions)
        simulation.context.setPeriodicBoxVectors(*self.init_boxes)
        simulation.context.setVelocitiesToTemperature(self.temp)

        # Init and append reporters
        SDR = StateDataReporter(**SDR_params)
        DCDR = DCDReporter(**DCDR_params)
        simulation.reporters.append(SDR)
        simulation.reporters.append(DCDR)

        # Run that bish ---> PAUSE -DC
        start = datetime.now()
        printf(f'Taking {n_steps_total} steps...')
        simulation.step(n_steps_total)
        printf(f'Simulation took {datetime.now() - start}')

        printf(f'Writing Final State to {self.state_fn}')
        self._write_state(simulation, self.state_fn)

    def image(self):
        traj = md.load(self.dcd_fn, top=self.sim_obs[3])
        traj.image_molecules()
        traj.save_dcd(self.dcd_fn)


    def assign_spring_attributes(self, intracellular_residue_indsA:np.array,
                                 intracellular_residue_indsB:np.array=None,
                                 spring_constant=83.68*spring_constant_unit):
        """
        
        """
        #If resids are not provided for B assume they are the same as A
        if intracellular_residue_indsB is None:
            intracellular_residue_indsB = intracellular_residue_indsA.copy()
        #Get selection strings for all atoms in the given resids lists
        selection_stringA = generate_selection_string(intracellular_residue_indsA)
        selection_stringB = generate_selection_string(intracellular_residue_indsA)
        
        my_args = dict(spring_centers1_pdb=self.pdb_fns["A"],
                       spring_centers2_pdb=self.pdb_fns["B"],
                       selection_1=selection_stringA,
                       selection_2=selection_stringB,
                       num_replicates=self.n_from_trailblazing)

        self.restraint_selection_string = selection_stringA
        self.spring_centers, indsA, indsB = make_interpolated_positions_array_from_selections(**my_args)
        self.spring_constant = spring_constant
        printf(f'Build spring centers with shape: {self.spring_centers.shape} and force constant {self.spring_constant}') 


    def _describe_state(self, sim: Simulation, name: str = "State"):
        """
        Report the energy of an openmm simulation

        Parameters:
            sim: Simulation: The OpenMM Simulation object to report the energy of
            name: string: Default="State" - An optional identifier to help distinguish what energy is being reported
        """
        state = sim.context.getState(getEnergy=True, getForces=True)
        self.PE = round(state.getPotentialEnergy()._value, 2)
        max_force = round(max(np.sqrt(v.x**2 + v.y**2 + v.z**2) for v in state.getForces()), 2)
        print(f"{name} has energy {self.PE} kJ/mol ", f"with maximum force {max_force} kJ/(mol nm)")
      
        
    def _write_state(self, sim: Simulation, xml_fn: str):
        """
        Serialize the State of an OpenMM Simulation to an XML file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to serialize the State of
            xml_fn: string: The path to the xmlfile to write the serialized State to
        """
        state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True, getEnergy=True)
        contents = XmlSerializer.serialize(state)
        with open(xml_fn, 'w') as f:
            f.write(contents)
        print(f'Wrote: {xml_fn}')
 
    
    def _write_system(self, sim: Simulation, xml_fn: str):
        """
        Serialize the System of an OpenMM Simulation to an XML file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to serialize the System of
            xml_fn: string: The path to the xmlfile to write the serialized System to
        """
        with open(xml_fn, 'w') as f:
            f.write(XmlSerializer.serialize(sim.system))
        print(f'Wrote: {xml_fn}')


    def _write_structure(self, sim: Simulation, pdb_fn: str):
        """
        Write the structure of an OpenMM Simulation to a PDB file.

        Parameters:
            sim: Simulation: The OpenMM Simulation to write the structure of
            pdb_fn: string: The path to the PDB file to write the structure to
        """
        with open(pdb_fn, 'w') as f:
            PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
        print(f'Wrote: {pdb_fn}')


if __name__ == '__main__':
    assert len(sys.argv) == 7, "You may have the wrong number of arguments :("
    args = dict(input_dir = sys.argv[1],
                centroid_A_name = sys.argv[2],
                centroid_B_name = sys.argv[3],
                save_dir = sys.argv[4],
                trailblazing_dcd_fn = sys.argv[5],
                replicate_index = int(sys.argv[6]))



    cb2_intracellular_inds = np.array([49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311])
    intra_inds_dict = {}
    
    for leg_iden in ["A", "B"]:
        if args[f"centroid_{leg_iden}_name"] in ["centroid_6", "centroid_13"]:
            #subtract 22 from these indices
            intra_inds_dict[leg_iden] = cb2_intracellular_inds - 22
        else:
            #subtract 23
            intra_inds_dict[leg_iden] = cb2_intracellular_inds - 23
    
    
    #Run Block
    UT = Straight_MD4US(**args)

    UT.assign_spring_attributes(intracellular_residue_indsA=intra_inds_dict["A"],
                                intracellular_residue_indsB=intra_inds_dict["B"])

    UT.run(sim_time=5*unit.nanosecond)

    UT.image()

    
