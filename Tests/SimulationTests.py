from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.DoMyCode import *
import os


class SimulationTests(TestCase):

    def setUp(self):
        self.cm = SimulationManager(os.path.expanduser("~/Desktop/Simulations"))