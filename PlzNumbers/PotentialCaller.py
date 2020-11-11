"""
Provides a Caller that Potential uses to actually evaluate the potential
"""

import numpy as np, os, multiprocessing as mp, sys, signal
from ..RynUtils import CLoader

__all__ = [
    "PotentialCaller"
]

class PotentialCaller:
    """
    Takes a pointer to a C++ potential calls this potential on a set of geometries / atoms / whatever
    """
    __props__ = [
        "bad_walker_file",
        "mpi_manager",
        "raw_array_potential",
        "vectorized_potential",
        "error_value",
        "fortran_potential",
        "transpose_call",
        "debug_print",
        "caller_retries"
    ]
    def __init__(self,
                 potential, *ignore,
                 bad_walker_file="bad_walkers.txt",
                 mpi_manager=None,
                 raw_array_potential = None,
                 vectorized_potential = False,
                 error_value = 10.e9,
                 fortran_potential=False,
                 transpose_call=None,
                 debug_print=False,
                 catch_abort=False,
                 caller_retries=1
                 ):
        if len(ignore) > 0:
            raise ValueError("Only one positional argument (for the potential) accepted")
        self.potential = potential
        self.bad_walkers_file = bad_walker_file
        self._mpi_manager = mpi_manager
        self.vectorized_potential = vectorized_potential
        self.raw_array_potential = fortran_potential if raw_array_potential is None else raw_array_potential
        self.error_value = error_value
        self._lib = None
        self._py_pot = not repr(self.potential).startswith("<capsule object ")  # wow this is a hack...
        self._wrapped_pot = None
        self.fortran_potential = fortran_potential
        if transpose_call is None:
            transpose_call = fortran_potential
        self.transpose_call = transpose_call
        self.debug_print = debug_print
        self.catch_abort=catch_abort
        self.caller_retries=caller_retries

    @classmethod
    def load_lib(cls):
        IRS_Ubuntu='2020.0.166'# needs to be synced with Dockerfile
        TBB_Ubutu='/opt/intel/compilers_and_libraries_{IRS}/linux/tbb/'.format(IRS=IRS_Ubuntu)
        IRS_CentOS = '2020.0.88'  # needs to be synced with Dockerfile
        TBB_CentOS='/opt/intel/compilers_and_libraries_{IRS}/linux/tbb/'.format(IRS=IRS_CentOS)
        loader = CLoader("PlzNumbers",
                         os.path.dirname(os.path.abspath(__file__)),
                         extra_compile_args=["-fopenmp", '-std=c++11'],
                         extra_link_args=["-fopenmp"],
                         include_dirs=[
                             "/lib/x86_64-linux-gnu",
                             os.path.join(TBB_Ubutu, "include"),
                             os.path.join(TBB_Ubutu, "lib", "intel64", "gcc4.8")
                         ],
                         runtime_dirs=[
                             "/lib/x86_64-linux-gnu",
                             os.path.join(TBB_Ubutu, "lib", "intel64", "gcc4.8")
                         ],
                         linked_libs=['tbb'],
                         source_files=["PlzNumbers.cpp", "Potators.cpp", "PyAllUp.cpp"]
                )
        return loader.load()

    @classmethod
    def reload(cls):
        try:
            os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PlzNumbers.so"))
        except OSError:
            pass
        return cls.load_lib()
    @property
    def lib(self):
        """

        :return:
        :rtype: module
        """
        if self._lib is None:
            self._lib = self.load_lib()
        return self._lib

    def call_single(self, walker, atoms, extra_bools=(), extra_ints=(), extra_floats=()):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """

        if self.catch_abort is not False or self.catch_abort is not None:
            if self.catch_abort is True:
                def handler(*args, **kwargs):
                    raise RuntimeError("Abort occurred?")
            elif isinstance(self.catch_abort, str) and self.catch_abort=="ignore":
                def handler(*args, **kwargs):
                    print("Abort occurred?")
            else:
                handler = self.catch_abort

            signal.signal(signal.SIGABRT, handler)

        if self._py_pot:
            return self.potential(walker, atoms, (extra_bools, extra_ints, extra_floats))
        else:
            if self.transpose_call:
                walker = walker.transpose()
            coords = np.ascontiguousarray(walker).astype(float)
            # print(coords)
            return self.lib.rynaLovesPoots(
                coords,
                atoms,
                self.potential,
                self.bad_walkers_file,
                float(self.error_value),
                bool(self.raw_array_potential),
                bool(self.debug_print),
                int(self.caller_retries),
                extra_bools,
                extra_ints,
                extra_floats
            )

    class PoolPotential:
        rooot_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        def __init__(self, potential, nprocs):
            self.nprocs = nprocs
            self.pot = potential
            try:
                self.pot_type = "dynamic"
                mod = sys.modules["DynamicImports." + potential.__module__]
            except KeyError:
                self.pot_type = "normal"
                mod = sys.modules[potential.__module__]

            self.pot_path = [os.path.dirname(mod.__file__), os.path.dirname(os.path.dirname(mod.__file__))]
            self._pool = None

        def __del__(self):
            self.terminate()

        def terminate(self):
            if self._pool is not None:
                self._pool.terminate()
        @property
        def pool(self):
            if self._pool is None:
                self._pool = self._get_pool()
            return self._pool

        def _init_pool(self, *path):
            import sys
            sys.path.extend(path)
            # print(sys.path)

        class FakePool:
            def __init__(self, ctx, pot, nprocs):
                """

                :param ctx:
                :type ctx: SpawnContext
                :param pot:
                :type pot:
                :param nprocs:
                :type nprocs:
                """
                self.arg_queue = ctx.Queue()
                self.res_queue = ctx.Queue()
                self.pot = pot
                self._started = False
                self.procs = [
                    ctx.Process(
                        target=self.drain_queue_and_call,
                        args=(self.arg_queue, self.res_queue, self.pot)
                    ) for i in range(nprocs)
                ]

            @staticmethod
            def drain_queue_and_call(arg_queue, res_queue, pot):
                """

                :param arg_queue:
                :type arg_queue: mp.Queue
                :param res_queue:
                :type res_queue: mp.Queue
                :param pot:
                :type pot: function
                :return:
                :rtype:
                """
                listening = True
                while listening:
                    block_num, data = arg_queue.get()
                    if block_num == -1:
                        break
                    try:
                        pot_val = pot(*data)
                    except:
                        res_queue.put((block_num, None))
                        raise
                    else:
                        res_queue.put((block_num, pot_val))

            def start(self):
                if not self._started:
                    for p in self.procs:
                        print("Starting : {}".format(p))
                        p.start()
                        print("Finished starting it...")
                    self._started = True

            def terminate(self):
                for p in self.procs:
                    self.arg_queue.put((-1, -1))
                    p.terminate()

            def map_pot(self, coords, atoms, extra):
                self.start()
                i=0
                for i, block in enumerate(coords):
                    self.arg_queue.put((i, (block, atoms, extra)))
                block_lens = i + 1
                blocks = [None]*block_lens
                for j in range(block_lens):
                    k, data = self.res_queue.get()
                    blocks[k] = data
                return blocks

        def _get_pool(self):

            if self.pot_type == "dynamic":
                # hack to handle the fact that our trial wavefunctions are dynamically loaded and pickle doesn't like it
                self._init_pool(self.rooot_dir, *self.pot_path)
                mod_spec = "DynamicImports." + self.pot.__module__
                mod = sys.modules[mod_spec]
                sys.modules[mod_spec.split(".", 1)[1]] = mod

            # it turns out Pool is managed using some kind of threaded interface so we'll try to work around that...
            np = mp.cpu_count()-1
            if np > self.nprocs:
                np = self.nprocs
            pool = self.FakePool(mp.get_context("spawn"), self.pot, np)
            return pool

        def call_pot(self, walkers, atoms, extra_bools=(), extra_ints=(), extra_floats=()):

            main_shape = walkers.shape[:-2]
            num_walkers = int(np.prod(main_shape))
            walkers = walkers.reshape((num_walkers,) + walkers.shape[-2:])
            blocks = np.array_split(walkers, min(self.nprocs, num_walkers))
            a = atoms
            e = (extra_bools, extra_ints, extra_floats)
            res = np.concatenate(self.pool.map_pot(blocks, a, e))
            return res.reshape(main_shape)

        def __call__(self, walkers, atoms, extra_bools=(), extra_ints=(), extra_floats=()):
            return self.call_pot(walkers, atoms, extra_bools=extra_bools, extra_ints=extra_ints,
                                 extra_floats=extra_floats)

    def _mp_wrap(self,
                 pot,
                 num_walkers,
                 mpi_manager
                 ):
        # We'll provide a wrapper that we can use with our functions to add parallelization
        # based on Pool.map
        # The wrapper will first check to make sure that we _are_ using a hybrid parallelization model

        from ..Interface import RynLib
        hybrid_p = RynLib.flags['multiprocessing']

        if hybrid_p:
            if mpi_manager is None:
                world_size = mp.cpu_count()
            else:
                hybrid_p = mpi_manager.hybrid_parallelization
                if hybrid_p:
                    world_size = mpi_manager.hybrid_world_size
                else:
                    world_size = 0 # should throw an error if we try to compute the block_size
        else:
            world_size = 0

        if hybrid_p:
            num_blocks = num_walkers if num_walkers < world_size else world_size
            potential = self.PoolPotential(pot, num_blocks)
        else:
            potential = pot

        return potential

    @property
    def mpi_manager(self):
        return self._mpi_manager
    @mpi_manager.setter
    def mpi_manager(self, m):
        self._mpi_manager = m

    def clean_up(self):
        if isinstance(self._wrapped_pot, self.PoolPotential):
            self._wrapped_pot.terminate()
            self._wrapped_pot = None
    def call_multiple(self, walker, atoms, extra_bools=(), extra_ints=(), extra_floats=()):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """

        smol_guy = walker.ndim == 3
        if smol_guy:
            walker = np.reshape(walker, (1,) + walker.shape[:1] + walker.shape[1:])

        if self._py_pot and self._wrapped_pot is None:
            num_walkers = int(np.product(walker.shape[:-2]))
            self._wrapped_pot = self._mp_wrap(self.potential, num_walkers, self.mpi_manager)
        if self._py_pot and self.mpi_manager is None:
            poots = self._wrapped_pot(walker, atoms, (extra_bools, extra_ints, extra_floats))
        elif self._py_pot:
            walker = walker.transpose((1, 0, 2, 3))
            if self.transpose_call:
                walker = walker.transpose((0, 1, 3, 2))
            coords = np.ascontiguousarray(walker).astype(float)
            poots = self.lib.rynaLovesPyPootsLots(
                coords,
                tuple(atoms),
                self._wrapped_pot,
                self.mpi_manager,
                (extra_bools, extra_ints, extra_floats)
            )
            if poots is not None:
                poots = poots.transpose()
        else:
            from ..Interface import RynLib
            omp = RynLib.flags['OpenMPThreads']
            tbb = RynLib.flags['TBBThreads']
            if omp and (self.mpi_manager is not None):
                omp = self.mpi_manager.hybrid_parallelization
            if tbb and (self.mpi_manager is not None):
                tbb = self.mpi_manager.hybrid_parallelization
            walker = walker.transpose((1, 0, 2, 3))
            if self.transpose_call:
                walker = walker.transpose((0, 1, 3, 2))
            coords = np.ascontiguousarray(walker).astype(float)
            poots = self.lib.rynaLovesPootsLots(
                coords,
                list(atoms), # turns out I use `PyList_GetTuple` in the code -_-
                self.potential,
                (extra_bools, extra_ints, extra_floats),
                self.bad_walkers_file,
                float(self.error_value),
                bool(self.raw_array_potential),
                bool(self.vectorized_potential),
                bool(self.debug_print),
                int(self.caller_retries),
                self.mpi_manager,
                bool(omp),
                bool(tbb)
            )
            if poots is not None:
                if self.mpi_manager is not None:
                    poots = poots.transpose()
                else:
                    shp = poots.shape
                    poots = poots.reshape(shp[1], shp[0]).transpose()
        if poots is not None and self.mpi_manager is not None:
            poots = poots[0] if smol_guy else poots
        return poots

    def __call__(self, walkers, atoms, *extra_args):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :return:
        :rtype:
        """

        extra_bools = []
        extra_ints = []
        extra_floats = []
        for a in extra_args:
            if a is True or a is False:
                extra_bools.append(a)
            elif isinstance(a, int):
                extra_ints.append(a)
            elif isinstance(a, float):
                extra_floats.append(a)

        if not isinstance(walkers, np.ndarray):
            walkers = np.array(walkers)

        ndim = walkers.ndim

        if ndim == 2:
            poots = self.call_single(walkers, atoms, extra_bools, extra_ints, extra_floats)
        elif ndim == 3 or ndim == 4:
            poots = self.call_multiple(walkers, atoms, extra_bools, extra_ints, extra_floats)
        else:
            raise ValueError(
                "{}: caller expects data of rank 2, 3, or 4. Got {}.".format(
                    type(self).__name__,
                    ndim
                    )
                )

        return poots