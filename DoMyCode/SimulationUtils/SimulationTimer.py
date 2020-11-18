import time as time

__all__ = [
    'SimulationTimer'
]

class SimulationTimer:
    """
    Super simple timer that does some formatting and shit
    """
    __props__ = [ "simulation_times" ]
    def __init__(self, simulation, simulation_times = None):
        self.simulation = simulation
        if simulation_times is None:
            simulation_times = []
        self.simulation_times = simulation_times
        self.start_time = None

    def start(self):
        self.start_time = time.time()
    def stop(self):
        if self.start_time is not None:
            self.simulation_times.append((self.start_time, time.time()))
            self.start_time = None
    @property
    def elapsed(self):
        if self.start_time is not None:
            return time.time() - self.start_time
        else:
            return 0