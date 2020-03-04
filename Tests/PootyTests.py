from Peeves.TestUtils import *
from unittest import TestCase
from PootyAndTheBlowfish.Templator import *
from PootyAndTheBlowfish.PotentialTemplator import PotentialTemplate
import sys

class PootyTests(TestCase):

    @inactiveTest
    def test_ApplyBaseTemplate(self):
        import os

        curdir = os.getcwd()
        template = os.path.join(curdir, "RynLib", "PootyAndTheBlowfish", "Templates", "PotentialTemplate")
        writer = TemplateWriter(template, LibName = "ploot")

        out = os.path.expanduser("~/Desktop")
        writer.iterate_write(out)

        worked = os.path.exists(os.path.join(out, "plootPot", "src", "CMakeLists.txt"))
        self.assertTrue(worked)

    @inactiveTest
    def test_SimplePotential(self):
        import os

        writer = PotentialTemplate(
            lib_name = "DumbPot",
            function_name = "DumbPot",
            linked_libs = [ "DumbPot" ],
            potential_source = TestManager.test_data("DumbPot"),
            requires_make = True
        )

        out = os.path.expanduser("~/Desktop")
        writer.apply(out)
