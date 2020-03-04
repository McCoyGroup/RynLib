from Peeves.TestUtils import *
from unittest import TestCase
from ConfigManager import *
import sys

class ConfigManagerTests(TestCase):

    @debugTest
    def test_SimpleConfig(self):
        import os

        cm = ConfigManager(os.path.expanduser("~/Desktop/Configs"))
        if "testDMC" not in cm.list_configs():
            print(cm.list_configs())
            cm.add_config("testDMC", walkers = 1000)
        cf = cm.load_config("testDMC")
        self.assertEquals(cf.walkers, 1000)