from Peeves.TestUtils import *
from unittest import TestCase
from ..Interface import RynLib, PotentialInterface

class EntosTests(TestCase):

    @debugTest
    def test_ConfigureEntos(self):
        PotentialInterface.configure_entos()

    @debugTest
    def test_EntosMPI(self):
        RynLib.test_MPI()