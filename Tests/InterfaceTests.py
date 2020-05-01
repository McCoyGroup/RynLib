from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.Interface import *
import os, shutil

class InterfaceTests(TestCase):

    @validationTest
    def test_LoadConfig(self):
        self.assertEquals(RynLib.root_directory(), "/tests")