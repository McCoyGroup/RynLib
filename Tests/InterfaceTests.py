from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.Interface import *
import os, shutil

class InterfaceTests(TestCase):

    @validationTest
    def test_LoadConfig(self):
        self.assertIsInstance(RynLib.root_directory(), str)