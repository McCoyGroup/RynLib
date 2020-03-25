from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.PlzNumbers import *
import os, shutil


class PotentialTests(TestCase):

    def setUp(self):
        self.dumb_pot = TestManager.test_data("DumbPot")
        self.pots_dir = os.path.expanduser("~/Desktop/potentials")

    def clear_cache(self):
        shutil.rmtree(self.pots_dir)
        try:
            os.remove("RynLib/PlzNumbers/PlzNumbers.so")
        except:
            pass

    @debugTest
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