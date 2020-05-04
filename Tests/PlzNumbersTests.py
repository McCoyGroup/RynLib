from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.PlzNumbers import *
from RynLib.Interface import *
import os, shutil


class PotentialTests(TestCase):

    def setUp(self):
        self.dumb_pot = TestManager.test_data("DumbPot")
        self.ho_pot = TestManager.test_data("HarmonicOscillator")
        self.lib_dumb_pot = TestManager.test_data("libdumbpot.so")
        self.pm = PotentialManager()

    def reset_lib(self):
        try:
            os.remove("RynLib/PlzNumbers/PlzNumbers.so")
        except:
            pass

    @validationTest
    def test_HarmonicOscillator(self):
        RynLib.test_HO()

    @validationTest
    def test_PythonHO(self):
        if 'PythonHO' not in self.pm.list_potentials():
            PotentialInterface.add_potential("PythonHO", src=TestManager.test_data("HOSimulation/HOPyPot"))
        py_ho = self.pm.test_potential("PythonHO")
        self.assertTrue(.0048 < py_ho < .0051)

    # @validationTest
    # def test_HarmonicOscillatorMPI(self):
    #     RynLib.test_ho_mpi()

    @validationTest
    def test_ConfigureEntos(self):
        if os.path.exists("/entos"):
            PotentialInterface.configure_entos()

    @validationTest
    def test_Entos(self):
        if os.path.exists("/entos"):
            RynLib.test_entos()