from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.Interface import *
import os, shutil


class InterfaceTests(TestCase):

    @debugTest
    def test_LoadConfig(self):
        self.assertEquals(GeneralConfig.get_conf().mpi_version, "3.1.4")

    @debugTest
    def test_CompileMPI(self):
        ...

