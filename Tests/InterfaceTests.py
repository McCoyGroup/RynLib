from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.Interface import *
import os, shutil


class InterfaceTests(TestCase):

    @debugTest
    def test_LoadConfig(self):
        self.assertEquals(RynLib.get_conf().mpi_version, "3.1.4")

    @debugTest
    def test_CompileMPI(self):
        RynLib.configure_mpi()

    @debugTest
    def test_CompileEntos(self):
        PotentialInterface.configure_entos()


