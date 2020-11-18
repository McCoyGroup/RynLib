import enum, numpy as np

__all__ = [
    "SimulationStepCounter"
]

class SimulationStepCounter:
    """
    A class that manages knowing what timestep it is and what that means we need to do
    """
    __props__ = [
        "step_num", "num_time_steps", "checkpoint_every", "gc_every",
        "equilibration_steps", "descendent_weight_every", "descendent_weighting_steps"
    ]
    def __init__(self,
                 sim,
                 step_num = 0,
                 num_time_steps = None,
                 checkpoint_every=None,
                 gc_every=None,
                 equilibration_steps=None,
                 descendent_weight_every = None,
                 descendent_weighting_steps = None
                 ):
        """
        :param step_num: the step number we're starting at
        :type step_num: int
        :param num_time_steps: the total number of timesteps we're running
        :type num_time_steps:
        :param checkpoint_every: how often to checkpoint the simulation
        :type checkpoint_every:
        :param equilibration_steps: the number of equilibration timesteps
        :type equilibration_steps: int
        :param descendent_weight_every: how often to calculate descendent weights
        :type descendent_weight_every: int
        :param descendent_weighting_steps: the number of steps taken in descendent weighting
        :type descendent_weighting_steps:
        """
        self.simulation = sim
        self.step_num = step_num
        self.num_time_steps = num_time_steps

        self._previous_checkpoint = step_num
        if isinstance(checkpoint_every, int):
            # bind a little lambda to check whether we've gone `checkpoint_every` steps further
            checkpoint_every = lambda s, c=checkpoint_every: s.step_num - s._previous_checkpoint >= c
        elif checkpoint_every is None:
            checkpoint_every = lambda *a: False
        self._checkpoint = checkpoint_every
        self._cached_checkpoint = None

        self._previous_gc = step_num
        if isinstance(gc_every, int):
            # bind a little lambda to check whether we've gone `checkpoint_every` steps further
            gc_every = lambda s, c=gc_every: s.step_num - s._previous_checkpoint >= c
        elif gc_every is None:
            gc_every = lambda *a: False
        self._do_gc = gc_every
        self._cached_gc = None

        self._equilibrated = False
        if isinstance(equilibration_steps, (int, np.integer)):
            equilibration_steps = lambda s, e=equilibration_steps: s.step_num > e  # classic lambda parameter binding
        self.equilibration_check = equilibration_steps

        self._dw_delay = descendent_weight_every
        self._dw_steps = descendent_weighting_steps
        self._last_dw_step = 0
        self._dw_initialized_step = None

    @property
    def checkpoint(self):
        if self._cached_checkpoint is None:
            self._cached_checkpoint = self._checkpoint(self)
            if self._cached_checkpoint:
                self._previous_checkpoint = self.step_num
        return self._cached_checkpoint

    @property
    def garbage_collect(self):
        if self._cached_gc is None:
            self._cached_gc = self._do_gc(self)
            if self._cached_gc:
                self._previous_gc = self.step_num
        return self._cached_gc

    def increment(self, n):
        self._cached_checkpoint = None
        self.step_num+=n

    @property
    def equilibrated(self):
        if not self._equilibrated:
            self._equilibrated = self.equilibration_check(self)
        return self._equilibrated

    @property
    def done(self):
        return self.step_num >= self.num_time_steps

    class DescendentWeightingStatus(enum.Enum):
        Waiting = "Waiting"
        Beginning = "Beginning"
        Complete = "Complete"
        Ongoing = "Ongoing"

    @property
    def descendent_weighting_status(self):
        """
        Keeps track of when descendent weighting started/was last done/whether we are currently doing it

        :return:
        :rtype:
        """
        step = self.step_num
        if self._dw_initialized_step is None:
            if step - self._last_dw_step >= self._dw_delay:
                status = self.DescendentWeightingStatus.Beginning
                self._dw_initialized_step = step
                self._dw_initialized_step = step
                self._last_dw_step = step
            else:
                status = self.DescendentWeightingStatus.Waiting
        elif step - self._dw_initialized_step >= self._dw_steps:
            status = self.DescendentWeightingStatus.Complete
            self._dw_initialized_step = None
        else:
            status = self.DescendentWeightingStatus.Ongoing
        return status
