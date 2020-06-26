from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.DoMyCode import *
from RynLib.Interface import *
from RynLib.PlzNumbers import PotentialManager
import os

class SimulationTests(TestCase):

    def setUp(self):
        self.sm = SimulationManager()
        self.im = ImportanceSamplerManager()
        self.pm = PotentialManager()

    @debugTest
    def test_SimpleHO(self):
        SimulationInterface.test_HO()
        f = self.sm.simulation_output_folder("test_HO")
        with open(os.path.join(f, "log.txt")) as out:
            out_stuff = out.read()
        clean_end = 'Ending simulation' in out_stuff and 'Zero-point Energy' in out_stuff
        if not clean_end:
            print(out_stuff)
        self.assertTrue(clean_end)
        sim = self.sm.load_simulation("test_HO")
        zpe = sim.analyzer.zpe
        self.assertTrue(1000 < zpe and zpe < 4000)
        SimulationInterface.archive_simulation("test_HO")

    @validationTest
    def test_SimpleHOMPI(self):
        if 'PythonHO' not in self.pm.list_potentials():
            PotentialInterface.add_potential("PythonHO", src=TestManager.test_data("HOSimulation/HOPyPot"))
        # if 'HarmonicOscillator' not in pm.list_potentials():
        #     PotentialInterface.configure_HO()
        sm = self.sm
        if "test_HO_mpi" in sm.list_simulations():
            sm.remove_simulation("test_HO_mpi")
        SimulationInterface.add_simulation("test_HO_mpi",
                           os.path.join(RynLib.test_data, "HOSimulation", "HOSimMPI")
                           )
        sm.run_simulation('test_HO_mpi')

        f = self.sm.simulation_output_folder("test_HO_mpi")
        with open(os.path.join(f, "log.txt")) as out:
            out_stuff = out.read()
        clean_end = 'Ending simulation' in out_stuff and 'Zero-point Energy' in out_stuff
        if not clean_end:
            print(out_stuff)
        self.assertTrue(clean_end)
        sim = self.sm.load_simulation("test_HO_mpi")
        zpe = sim.analyzer.zpe
        self.assertTrue(1000 < zpe and zpe < 4000)
        SimulationInterface.archive_simulation("test_HO_mpi")

    @debugTest
    def test_SimpleHOPy(self):
        if 'PythonHO' not in self.pm.list_potentials():
            PotentialInterface.add_potential("PythonHO", src=TestManager.test_data("HOSimulation/HOPyPot"))
        if 'test_HO_py' in self.sm.list_simulations():
            self.sm.archive_simulation('test_HO_py')
        SimulationInterface.add_simulation("test_HO_py", src=TestManager.test_data("HOSimulation/HOSimPy"))
        self.sm.run_simulation('test_HO_py')
        f = self.sm.simulation_output_folder("test_HO_py")
        with open(os.path.join(f, "log.txt")) as out:
            out_stuff = out.read()
        clean_end = 'Ending simulation' in out_stuff and 'Zero-point Energy' in out_stuff
        if not clean_end:
            print(out_stuff)
        self.assertTrue(clean_end)

    @debugTest
    def test_SimpleHOImp(self):
        SimulationInterface.test_HO_imp()
        f = self.sm.simulation_output_folder("test_HO_imp")
        with open(os.path.join(f, "log.txt")) as out:
            out_stuff = out.read()
        self.assertTrue('Ending simulation' in out_stuff and 'Zero-point Energy' in out_stuff)
        SimulationInterface.archive_simulation("test_HO_imp")

    @validationTest
    def test_ImportanceSampling(self):
        if "HOSampler" in self.im.list_samplers():
            self.im.remove_sampler("HOSampler")
        SimulationInterface.add_sampler(
            "HOSampler",
            source=os.path.join(RynLib.test_data, "HOSimulation", "HOTrialWavefunction")
        )
        self.im.test_sampler("HOSampler")

    # @validationTest
    # def test_SimulationArchiving(self):
    #     SimulationInterface.test_add_HO()
    #     SimulationInterface.archive_simulation("test_HO")
    #     self.assertTrue(len(SimulationManager().list_archive()) > 0 )
