"""
The SimulationLoader that I have here is basically entirely _legacy_ code.
The SimulationManager interface should be used in its place, but I'm keeping it here until that interface is solidified
"""



import numpy as np
from .DoMyCode import *
import argparse

class SimulationLoader:
    """A little class that takes arguments off the command line and uses them to initialize a DMC"""
    def __init__(self, description, potential, walkers, **opts):
        self._parser = None
        self.params = None
        self.description = description
        self.potential = potential
        self.walkers = walkers
        self.opts = opts
        self._loaded = False

    @property
    def parser(self):
        if self._parser is None:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--name",
                default = "mpi_dmc",
                type = str,
                dest = 'name'
            )
            parser.add_argument(
                "--walkers_per_core",
                default = 2,
                type = int,
                dest = 'walkers_per_core'
            )
            parser.add_argument(
                "--steps_per_call",
                default = 10,
                type = int,
                dest = 'steps_per_call'
            )
            parser.add_argument(
                "--equilibration",
                default = 1000,
                type = int,
                dest = 'equilibration'
            )
            parser.add_argument(
                "--total_time",
                default = 10000,
                type = int,
                dest = 'total_time'
            )
            parser.add_argument(
                "--descendent_weighting_delay",
                default = 500,
                type = int,
                dest = 'descendent_weighting_delay'
            )
            parser.add_argument(
                "--descendent_weighting_steps",
                default = 50,
                type = int,
                dest = 'descendent_weighting_steps'
            )
            parser.add_argument(
                "--delta_tau",
                default = 5.0,
                type = float,
                dest = 'delta_tau'
            )
            parser.add_argument(
                "--checkpoint_every",
                default = 100,
                type = int,
                dest = 'checkpoint_every'
            )
            parser.add_argument(
                "--debug_level",
                default = Simulation.LOG_DEBUG,
                type = int,
                dest = 'debug_level'
            )
            parser.add_argument(
                "--write_wavefunctions",
                default = True,
                type = bool,
                dest = "write_wavefunctions"
            )
            self._parser = parser
        return self._parser

    def load_params(self):
        if not self._loaded:
            self.params = self.parser.parse_args()
            pdict = dict(**self.params)
            pdict.update(
                write_wavefunction = self.params.write_wavefunction,
                debug_level = self.params.debug_level,
                checkpoint_every = self.params.checkpoint_every,
                delta_tau = self.params.delta_tau,
                descendent_weighting_steps = self.params.descendent_weighting_steps,
                descendent_weighting_delay = self.params.descendent_weighting_delay,
                total_time = self.params.total_time,
                equilibration = self.params.equilibration,
                walkers_per_core = self.params.walkers_per_core,
                steps_per_call = self.params.steps_per_call,
                name = self.params.name
            )
            pdict.update(**self.opts)

            self.opts = pdict
            self._loaded = True

    def get_simulation(self, **options):
        self.load_params()

        opts = dict(
            potential = self.potential,
            description = self.description,
            walker_set = self.walkers,
            **self.opts
            )
        opts.update(**options)

        #
        # Actual run parameters
        #
        dwDelay = opts["descendent_weighting_delay"]
        nDw = opts["descendent_weighting_steps"]
        deltaT = opts["delta_tau"]
        alpha = 1.0 / (2.0 * deltaT)
        equil = opts["equilibration"]
        ntimeSteps = opts["total_time"]
        ntimeSteps += nDw
        propSteps = opts["steps_per_call"]
        checkpointAt = opts["checkpoint_every"]
        name = opts["name"]
        debug_level = opts["debug_level"]
        write_wavefunctions = opts["write_wavefunctions"]
        descendent_weighting = (dwDelay, nDw)
        D = 1/2.0
        walker_set = opts["walker_set"]
        potential = opts["potential"]
        description = opts["description"]

        for k in (
            "descendent_weighting_delay", "descendent_weighting_steps",
            "delta_tau", "equilibration", "total_time", "steps_per_call",
            "checkpoint_every", "name", "debug_level", "write_wavefunctions",
            "walker_set", "potential", "description", "verbosity"
        ):
            if k in opts:
                del opts[k]

        return Simulation(
            name,
            description,
            D = D,
            walker_set = walker_set,
            time_step = deltaT,
            alpha = alpha,
            potential = potential,
            num_time_steps = ntimeSteps,
            steps_per_propagation = propSteps,
            equilibration = equil,
            checkpoint_at = checkpointAt,
            descendent_weighting = descendent_weighting,
            write_wavefunctions = write_wavefunctions,
            verbosity = debug_level,
            **opts
        )