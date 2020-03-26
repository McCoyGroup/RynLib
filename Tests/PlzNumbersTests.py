from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.PlzNumbers import *
from RynLib.Interface import RynLib
import os, shutil


class PotentialTests(TestCase):

    def setUp(self):
        self.dumb_pot = TestManager.test_data("DumbPot")
        self.ho_pot = TestManager.test_data("HarmonicOscillator")
        self.lib_dumb_pot = TestManager.test_data("libdumbpot.so")
        self.pots_dir = RynLib.get_conf().potential_directory#os.path.expanduser("~/Desktop/potentials")

    def clear_cache(self):
        shutil.rmtree(self.pots_dir)

    def reset_lib(self):
        try:
            os.remove("RynLib/PlzNumbers/PlzNumbers.so")
        except:
            pass

    @validationTest
    def test_LoadDumbPot(self):

        pot = Potential(
            "DumbPot",
            self.dumb_pot,
            wrap_potential=True,
            function_name='DumbPot',
            requires_make=True,
            linked_libs=['DumbPot'],
            potential_directory=self.pots_dir
        )

        self.assertEquals(
            pot.caller([[0, 0, 0], [1, 1, 1]], ["H", "H"]),
            71.5
        )

    @inactiveTest
    def test_LoadLibDumbPot(self):

        pot = Potential(
            "DumbPot2",
            self.lib_dumb_pot,
            wrap_potential=True,
            function_name='DumbPot',
            requires_make=False,
            linked_libs=['DumbPot'],
            potential_directory=self.pots_dir
        )

        self.assertEquals(
            pot.caller([[0, 0, 0], [1, 1, 1]], ["H", "H"]),
            71.5
        )

    @validationTest
    def test_HarmonicOscillator(self):
        import numpy as np

        pot = Potential(
            "HarmonicOscillator",
            self.ho_pot,
            wrap_potential=True,
            function_name='HarmonicOscillator',
            arguments=(('re', float), ('k', float)),
            requires_make=True,
            linked_libs=['HarmonicOscillator'],
            potential_directory=self.pots_dir
        )

        self.assertAlmostEquals(
            pot.caller([[1, 2, 3], [1, 1, 1]], ["H", "H"], .9, 1.),
            1/2*(np.linalg.norm(np.array([1, 2, 3])-np.array([1, 1, 1])) - .9)**2
        )