from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.DoMyCode import *
from RynLib.Interface import *
import os


class SimulationTests(TestCase):

    def setUp(self):
        self.cm = SimulationManager(GeneralConfig.get_conf().simulations_directory)

    @debugTest
    def test_SimpleHO(self):
        ...

    @debugTest
    def test_ImportanceSampling(self):
        ...