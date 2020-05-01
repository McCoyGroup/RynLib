
config = dict(
    walkers=dict(
            atoms=["H", "H"],
            initial_walker=
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ],
            num_walkers=100
        ),
    time_step=1,
    steps_per_propagation=5,
    parameters=[.9, 1.0]
)
