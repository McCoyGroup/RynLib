
config = dict(
    name="test_HO",
    description="A sample HO simulation",
    potential=dict(
        name="HarmonicOscillator",
        parameters=[.9, 1.0]#re and k
    ),
    walker_set=dict(
            atoms=["H", "H"],
            initial_walker=[[1, 0, 0], [0, 0, 0]],
            walkers_per_core=1000
        ),
    steps_per_propagation=10,
    num_time_steps=10000,
    checkpoint_at=100,
    equilibration_steps=1000,
    descendent_weight_every=500,
    descendent_weighting_steps=50
)