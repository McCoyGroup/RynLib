import numpy as np
from ...RynUtils import Constants

class SimulationAnalyzer:
    __props__ = [ "zpe_averages" ]
    def __init__(self, simulation, zpe_averages = 5000):
        self.sim = simulation
        self.zpe_averages = zpe_averages

    @property
    def zpe(self):
        return self.get_zpe()

    def get_zpe(self, n=None):
        import itertools
        if n is None:
            n = self.zpe_averages
        if len(self.sim.reference_potentials) > n:
            vrefs = list(
                itertools.islice(
                    self.sim.reference_potentials,
                    len(self.sim.reference_potentials) - n,
                    None,
                    1
                )
            )
        else:
            vrefs = self.sim.reference_potentials
        return Constants.convert(np.average(np.array(vrefs)), "wavenumbers", in_AU=False)

    class Plotter:
        _mpl_loaded = False

        @classmethod
        def load_mpl(cls):
            if not cls._mpl_loaded:
                import matplotlib as mpl
                # mpl.use('Agg')
                cls._mpl_loaded = True

        @classmethod
        def plot_vref(cls, sim):
            """

            :param sim:
            :type sim: Simulation
            :return:
            :rtype:
            """

            import matplotlib.pyplot as plt
            e = np.array(sim.reference_potentials)
            n = np.arange(len(e))
            fig, axes = plt.subplots()
            e = Constants.convert(e, 'wavenumbers', in_AU=False)
            axes.plot(n, e)
            # axes.set_ylim([-3000,3000])
            plt.show()

        @classmethod
        def plot_psi(cls, sim):
            """

            :param sim:
            :type sim: Simulation
            :return:
            :rtype:
            """
            # assumes 1D psi...
            import matplotlib.pyplot as plt
            w = sim.walkers
            fig, axes = plt.subplots()

            hist, bins = np.histogram(w.coords.flatten(), weights=(w.weights), bins=20, density=True)
            bins -= (bins[1] - bins[0]) / 2
            axes.plot(bins[:-1], hist)
            plt.show()

        @classmethod
        def plot_psi2(cls, sim):
            """

            :param sim:
            :type sim: Simulation
            :return:
            :rtype:
            """
            # assumes 1D psi...
            import matplotlib.pyplot as plt
            w = sim.walkers
            fig, axes = plt.subplots()
            coord, dw, ow = sim.wavefunctions[-1]
            coord = coord.flatten()

            hist, bins = np.histogram(coord, weights=dw, bins=20, density=True)
            bins -= (bins[1] - bins[0]) / 2
            axes.plot(bins[:-1], hist)
            plt.show()