from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.DoMyCode import *
from RynLib.Interface import *
import os

class SimulationTests(TestCase):

    def setUp(self):
        self.cm = SimulationManager()
        self.im = ImportanceSamplerManager()

    @debugTest
    def test_SimpleHO(self):
        SimulationInterface.test_HO()

    @debugTest
    def test_ImportanceSampling(self):
        SimulationInterface.add_sampler(
            "HOSampler",
            source=os.path.join(RynLib.test_data, "HOSimulation", "HOTrialWavefunction")
        )
        self.im.test_sampler("HOSampler")
