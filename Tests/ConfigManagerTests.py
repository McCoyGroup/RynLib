from Peeves.TestUtils import *
from unittest import TestCase
from RynLib.Interface import RynLib
from RynLib.RynUtils.ConfigManager import *
import os

class ConfigManagerTests(TestCase):

    def setUp(self):
        self.cm = ConfigManager(os.path.join(RynLib.root_directory(), "configs"))

    def ensure_config(self):
        if "testDMC" not in self.cm.list_configs():
            self.cm.add_config("testDMC", walkers=1000)

    @validationTest
    def test_AddConfig(self):
        cm = self.cm
        self.ensure_config()
        cf = cm.load_config("testDMC")
        self.assertEquals(cf.walkers, 1000)

    @validationTest
    def test_RemoveConfig(self):

        cm = self.cm
        self.ensure_config()
        if "testDMC" in cm.list_configs():
            cm.remove_config("testDMC")

        self.assertTrue("testDMC" not in cm.list_configs())

    @validationTest
    def test_EditConfig(self):

        self.ensure_config()
        self.cm.edit_config("testDMC", walkers = 3000)
        cf = self.cm.load_config("testDMC")

        self.assertEquals(cf.walkers, 3000)

        self.cm.edit_config("testDMC", walkers=1000)