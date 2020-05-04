
config = dict(
    name="test_HO_python",
    description="A sample HO simulation using python Harmonic Oscillator",
    potential=dict(
        name="PythonHO",
        parameters=[.9, 1.0]#re and k
    ),
    walker_set=dict(
            atoms=["H", "H"],
            initial_walker=[[1, 0, 0], [0, 0, 0]],
            walkers_per_core=100
        ),
    mpi_manager = None,
    time_step=1,
    steps_per_propagation=3,
    num_time_steps=120,
    checkpoint_every=6,
    equilibration_steps=30,
    descendent_weight_every=30,
    descendent_weighting_steps=9
)