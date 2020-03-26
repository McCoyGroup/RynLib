
config = dict(
    name="water_3000",
    description="A 3000 walker water simulation",
    potential="entos",
    walker_set=dict(
            atoms=["O", "H", "H"],
            initial_walker=[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
            walkers_per_core=8
        ),
    # alpha = None,
    steps_per_propagation=10,
    # trial_wvfn = None,
    num_time_steps=10000,
    checkpoint_at=100,
    equilibration_steps=1000,
    descendent_weight_every=500,
    descendent_weighting_steps=50
)