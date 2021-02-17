"""
Provides a Caller that Potential uses to actually evaluate the potential
"""

import numpy as np, os, multiprocessing as mp, sys
from ..RynUtils import CLoader

from .PotentialArguments import PotentialArgumentHolder
from .FFI import FFIModule, FFIMethod

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
                 potential,
                 function_name,
                 *ignore,
                 mpi_manager=None,
                 bad_walker_file='',
                 raw_array_potential=None,
                 vectorized_potential=None,
                 error_value=1.0e9,
                 fortran_potential=False,
                 transpose_call=None,
                 debug_print=False,
                 catch_abort=False,
                 caller_retries=0
                 ):
        if len(ignore) > 0:
            raise ValueError("Only two positional arguments accepted (module and method name)")
        if isinstance(potential, FFIModule):
            self.module = potential
            self.potential = potential.get_method(function_name)
            self._py_pot = False
            self._caller_api = 2
            self.vectorized_potential = self.potential.vectorized
        elif repr(potential).startswith("<capsule object "):
            self.module = None
            self.potential = potential
            self._py_pot = False
            self._caller_api = 1
            self.vectorized_potential = bool(vectorized_potential)
        else:
            self.module = None
            self.potential = potential
            self._py_pot = True
            self._caller_api = 1
            self.vectorized_potential = False
        self.function_name = function_name

        self._mpi_manager = mpi_manager
        self._lib = None

        self._wrapped_pot = None

        # flags that we use
        self.bad_walkers_file = bad_walker_file
        # self.vectorized_potential = vectorized_potential
        self.raw_array_potential = fortran_potential if raw_array_potential is None else raw_array_potential
        self.error_value = error_value
        self.fortran_potential = fortran_potential
        if transpose_call is None:
            transpose_call = fortran_potential
        self.transpose_call = transpose_call
        self.debug_print = debug_print
        self.catch_abort = catch_abort
        self.caller_retries = caller_retries


    libs_folder = os.path.join(os.path.dirname(__file__), "libs")
    cpp_std = '-std=c++17'
    IRS_Ubuntu = '2020.0.166'  # needs to be synced with Dockerfile
    TBB_Ubutu = '/opt/intel/compilers_and_libraries_{IRS}/linux/tbb/'.format(IRS=IRS_Ubuntu)
    IRS_CentOS = '2020.0.88'  # needs to be synced with Dockerfile
    TBB_CentOS = '/opt/intel/compilers_and_libraries_{IRS}/linux/tbb/'.format(IRS=IRS_CentOS)
    @classmethod
    def load_lib(cls):
        loader = CLoader("PlzNumbers",
                         os.path.dirname(os.path.abspath(__file__)),
                         extra_compile_args=["-fopenmp", cls.cpp_std],
                         extra_link_args=["-fopenmp"],
                         include_dirs=[
                             "/lib/x86_64-linux-gnu",
                             os.path.join(cls.TBB_Ubutu, "include"),
                             os.path.join(cls.TBB_Ubutu, "lib", "intel64", "gcc4.8"),
                             os.path.join(cls.libs_folder),
                             np.get_include()
                         ],
                         runtime_dirs=[
                             "/lib/x86_64-linux-gnu",
                             os.path.join(cls.TBB_Ubutu, "lib", "intel64", "gcc4.8")
                         ],
                         linked_libs=["plzffi"
                            , 'tbb', 'tbbmalloc', 'tbbmalloc_proxy'
                          ],
                         source_files=[
                             # "PyAllUp.cpp",
                             "PlzNumbers.cpp",
                             "PotentialCaller.cpp",
                             "MPIManager.cpp",
                             "CoordsManager.cpp",
                             "PotValsManager.cpp",
                             "ThreadingHandler.cpp"
                         ],
                         macros=[("_TBB", True)],
                         requires_make=True
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

    def _get_omp_threads(self):
        from ..Interface import RynLib
        return RynLib.flags['OpenMPThreads']

    def _get_tbb_threads(self):
        from ..Interface import RynLib
        return RynLib.flags['TBBThreads']

    @property
    def caller_api_version(self):
        return self._caller_api

    class CallerParameters:
        arg_sig = "iOOOdpippppp"
        def __init__(self,
                     potential,
                     args,
                     _function_name=None,
                     _caller_api=0,
                     _bad_walkers_file=None,
                     _error_value=None,
                     _raw_array_potential=None,
                     _vectorized_potential=None,
                     _debug_print=None,
                     _caller_retries=None,
                     _use_openmp=None,
                     _use_tbb=None,
                     _python_potential=None
                     ):
            if not isinstance(args, PotentialArgumentHolder):
                raise TypeError("{} expects args to be of type {}".format(
                    type(self).__name__,
                    PotentialArgumentHolder.__name__
                ))

            self.args=args
            self.caller_api=_caller_api
            self.function_name=_function_name
            self.bad_walkers_file=_bad_walkers_file
            self.error_value=_error_value
            self.raw_array_potential=_raw_array_potential
            self.vectorized_potential=_vectorized_potential
            self.debug_print=_debug_print
            self.caller_retries=_caller_retries
            self.use_openmp=_use_openmp
            self.use_tbb=_use_tbb
            self.python_potential=_python_potential

        arg_keys = (
                'caller_api',
                'function_name',
                'args',
                'bad_walkers_file',
                'error_value',
                'debug_print',
                'caller_retries',
                'raw_array_potential',
                'vectorized_potential',
                'use_openmp',
                'use_tbb',
                'python_potential'
            )
        @property
        def argvec(self):
            args = []
            for k in self.arg_keys:
                v = getattr(self, k)
                if v is None:
                    raise ValueError("{} doesn't have a value for {}".format(
                        type(self).__name__,
                        k
                    ))
                args.append(v)
            return tuple(args)

        def __repr__(self):
            return "{}{}".format("CallerParameters", str(self.argvec))

    def call_multiple(self, walker, atoms, extra_args,
                      omp_threads=None,
                      tbb_threads=None
                      ):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :param extra_args:
        :type extra_args: PotentialArguments
        :return:
        :rtype:
        """

        smol_guy = walker.ndim == 2
        if smol_guy:
            walker = np.reshape(walker, (1, 1) + walker.shape[:1] + walker.shape[1:])
        else:
            smol_guy = walker.ndim == 3
            if smol_guy:
                walker = np.reshape(walker, (1,) + walker.shape[:1] + walker.shape[1:])

        if self._py_pot and self.mpi_manager is None:
            poots = self.potential(walker, atoms, extra_args)
        else:
            # clumy way to determine the # of threads we need
            omp = self._get_omp_threads() if omp_threads is None else omp_threads
            if omp:
                if (self.mpi_manager is not None):
                    omp = self.mpi_manager.hybrid_parallelization
                tbb = False
            else:
                omp = False
                tbb = self._get_tbb_threads() if tbb_threads is None else tbb_threads
                if tbb and (self.mpi_manager is not None):
                    tbb = self.mpi_manager.hybrid_parallelization

            # for MPI efficiency, we reshape so we have # walkers, then # calls
            # realistically this doesn't matter, but it's here for historical reasons...
            walker = walker.transpose((1, 0, 2, 3))
            if self.transpose_call:
                walker = walker.transpose((0, 1, 3, 2))
            coords = np.ascontiguousarray(walker).astype(float)
            # gotta comment this out or we end up reversing the effect of the transpose
            # if self.transpose_call:
            #     coords = np.asfortranarray(coords)
            new_style = isinstance(self.potential, (FFIModule, FFIMethod))
            # do the actual call into the C++ side
            poots = self.lib.rynaLovesPootsLots(
                bool(self.debug_print),
                coords,
                atoms,
                self.module.captup if new_style else self.potential,
                self.CallerParameters(
                    self.potential,
                    extra_args,
                    _function_name=self.function_name if new_style else '_potential',
                    _caller_api=self.caller_api_version,
                    _bad_walkers_file=self.bad_walkers_file,
                    _error_value=float(self.error_value),
                    _raw_array_potential=bool(self.raw_array_potential),
                    _vectorized_potential=bool(self.vectorized_potential),
                    _debug_print=bool(self.debug_print),
                    _caller_retries=int(self.caller_retries),
                    _use_openmp=bool(omp),
                    _use_tbb=bool(tbb),
                    _python_potential=self._py_pot
                ),
                self.mpi_manager
            )
            if poots is not None:
                if self.mpi_manager is not None: # switch the shape back around
                    poots = poots.transpose()
                else:
                    shp = poots.shape
                    poots = poots.reshape(shp[1], shp[0]).transpose()

        if poots is not None and smol_guy:# and self.mpi_manager is not None:
            poots = poots.squeeze()
        return poots

    def __call__(self, walkers, atoms, extra_args):
        """

        :param walker:
        :type walker: np.ndarray
        :param atoms:
        :type atoms: List[str]
        :param extra_args:
        :type extra_args: PotentialArguments
        :return:
        :rtype:
        """

        if not isinstance(walkers, np.ndarray):
            walkers = np.array(walkers)

        ndim = walkers.ndim

        if 1 < ndim < 5:
            poots = self.call_multiple(walkers, atoms, extra_args)
        else:
            raise ValueError(
                "{}: caller expects data of rank 2, 3, or 4. Got {}.".format(
                    type(self).__name__,
                    ndim
                    )
                )

        return poots

    class PoolPotential:
        rooot_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        def __init__(self, potential, nprocs):
            raise DeprecationWarning("this was a failed experiment")
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
        raise DeprecationWarning("failed experiment")
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