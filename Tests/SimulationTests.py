from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.DoMyCode import *
from RynLib.Interface import *
import os


class SimulationTests(TestCase):

    def setUp(self):
        self.cm = SimulationManager(RynLib.get_conf().simulations_directory)

    @debugTest
    def test_SimpleHO(self):
        ...

    @debugTest
    def test_ImportanceSampling(self):
        # Write a set of unit tests that checks if, given a certain trial wavefunction, the
        # importance sampler class will not throw errors when calling `accept`, `metropolis`, etc.
        ...
