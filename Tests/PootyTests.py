from Peeves.TestUtils import *
from unittest import TestCase
from PootyAndTheBlowfish.Templator import *
import sys

class PootyTests(TestCase):

    @debugTest
    def test_ApplyBaseTemplate(self):
        import os

        curdir = os.getcwd()
        template = os.path.join(curdir, "RynLib", "PootyAndTheBlowfish", "template")
        writer = TemplateWriter(template, LibName = "ploot")

        out = os.path.expanduser("~/Desktop")
        writer.iterate_write(out)

        worked = os.path.exists(os.path.join(out, "plootPot", "src", "CMakeLists.txt"))
        self.assertTrue(worked)
