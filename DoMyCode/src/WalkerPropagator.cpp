//
// Created by Mark Boyer on 1/24/20.
//

/*
 * Targeted, asynchronous MPI methods for walker broadcasting and whatnot
 *
 * We'll make use of a core-specific CORE_PROPAGATOR which will keep track of which walkers are on that given node
 *
 * On each loop we'll ask for the next n potential values, do the branching on the python side, send out a flag to all
 * the nodes telling them whether or not they need to (a) send their walker (b) update their walker or (c) do nothing
 * the ones that need to send will send at then these will be sent out to the ones that need to recieve
 *
 */

WalkerPropagator CORE_PROPAGATOR; // This is the specific WalkerPropagator we're gonna be using for a given core
void _mpiSetUpSimulation(
        RawWalkerBuffer raw_data,
        const Names &atoms,
        const Weights &sigmas,
        int ncalls,
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
        ) {

    int num_walkers_per_core = (num_walkers / world_size); // we're gonna assume the former is divisible by the latter

    // create a buffer for the walkers to be fed into MPI
    int walker_size = num_atoms*3*sizeof(Real_t);
    auto walker_buf = (RawWalkerBuffer) malloc(num_walkers_per_core*walker_size);

    // Scatter data buffer to processors
    MPI_Scatter(
            raw_data,  // raw data buffer to chunk up
            num_walkers_per_core*walker_size, // three coordinates per atom per num_atoms per walker
            MPI_DOUBLE, // coordinates stored as doubles
            walker_buf, // raw array to write into
            num_walkers_per_core*walker_size, // single energy
            MPI_DOUBLE, // energy returned as doubles
            0, // root caller
            MPI_COMM_WORLD // communicator handle
    );

    // I might be using too much memory here? But ah well
    Configurations walkers (num_walkers_per_core, Coordinates(num_atoms, Point(3)));
    for (int n = 0; n < num_walkers_per_core; n++){
        walkers[n] = _getWalkerCoords(walker_buf, n, num_atoms);
    }
    free(walker_buf);

    // On each core parse these back out, allocate them into a Configurations array, init the CORE_PROPAGATOR
    CORE_PROPAGATOR.init(
            atoms,
            walkers,
            sigmas,
            ncalls,
            world_size
            );

    // https://stackoverflow.com/a/48168458/5720002
    // If MPI_Scatter couldn't promise us a specific type of ordering we'd need to do some work here to get the
    // appropriate ordering out. Instead we can just make use of that fact to compute the initial ordering once on the
    // python size

}

RawWalkerBuffer _mpiGetWalkersFromCores(
        Py_ssize_t num_walkers,
        Py_ssize_t num_atoms,
        int world_size,
        int world_rank
) {

    // We'll use this as a way to get _all_ of the walker data out of the simulation
    // it won't be used much, I imagine, but as long as it gets called on all of the cores it should work...
    // maybe every like 100th timestep or something we'll get back all of our walker data, you know?

    int walker_size = num_atoms*3*sizeof(Real_t);
    int num_walkers_per_core = num_walkers / world_size;
    RawWalkerBuffer walker_buf;
    if (world_rank == 0) {
        // allocate buffer for getting the walkers back out of the simulation
        walker_buf = (RawWalkerBuffer) malloc(num_walkers*walker_size);
    }

    MPI_Gather(
            CORE_PROPAGATOR.walkers.data(),
            num_walkers_per_core*walker_size, // number of walkers fed in
            MPI_DOUBLE,
            walker_buf, // buffer to get the potential values back
            num_walkers_per_core*walker_size, // number of walkers fed in
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
    );

    return walker_buf;

}

RawPotentialBuffer _mpiGetPotFromCores(
        Py_ssize_t num_walkers,
        int world_size,
        int world_rank
) {
    // this'll have to be called by every core so that the Gather actually works

    int num_walkers_per_core = num_walkers / world_size;
    RawPotentialBuffer pot_buf;
    if (world_rank == 0) {
        // allocate buffer for getting the potential values back out of the simulation
        auto pot_buf = (RawPotentialBuffer) malloc(num_walkers);
    }

    MPI_Gather(
            CORE_PROPAGATOR.get_pots(),
            CORE_PROPAGATOR.prop_steps * num_walkers_per_core, // number of steps calculated
            MPI_DOUBLE,
            pot_buf, // buffer to get the potential values back
            CORE_PROPAGATOR.prop_steps * num_walkers_per_core, // number of walkers fed in
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
    );

    return pot_buf;

}

Configurations _mpiRemapWalkersToCores(
        std::vector<WalkerIDs> reborn_from_the_ashes,
        int world_size,
        int world_rank
        ) {

    // New thought:
    //  we send async broadcast a message that encodes how many walkers to expect to get and how many to expect to send
    //  this will allow ones that get 00 to continue forward and HF away
    //
    //  those that are told they need to get walkers will then get a message encoding which walkers they can expect to update and from where
    //  that will be implicit in the second digit in the first message and thus will allow them to set up the appropriate Irecv calls
    //
    //  after that message goes out the ones that will need to send will be sent basically the same message, but telling them which
    //  to send and to where and they'll push out the appropriate Isend calls
    //
    //  I think this maximizes asynchronicity since only the cores that need to chat will ever be frozen while they wait for their messages to go through
    //  (and actually only the ones that need to _recieve_ will ever have to wait)


    // here I'm making use of the fact that CoreID and WalkerID have to be the same type (int) but it does feel icky -_-
    std::vector<WalkerIDs> messages(CORE_PROPAGATOR.world_size, WalkerIDs(2, 0));
    std::vector<WalkerIDs> birthing(CORE_PROPAGATOR.world_size);
    std::vector<WalkerIDs> dying(CORE_PROPAGATOR.world_size);
    if (world_rank == 0) {
        CoreID from_core;
        CoreID to_core;
        for (auto walkers : reborn_from_the_ashes) {
            from_core = CORE_PROPAGATOR.walker_map[walkers[0]];
            messages[from_core][0]++;
            birthing[from_core].push_back(walkers[0]);
            to_core = CORE_PROPAGATOR.walker_map[walkers[1]];
            messages[to_core][1]++;
            dying[from_core].push_back(walkers[1]);
        }
    }

    // send out the messages and capture them in message
    WalkerIDs message(2, 0);
    MPI_Scatter( // I might want this to be Iscatter ?
            messages.data(),
            2,
            MPI_INT,
            message.data(),
            2,
            MPI_INT,
            0,
            MPI_COMM_WORLD
            );

    // now if we're recieving children set up a Recv to get them
    std::vector<MPI_Request> reqs;
    Configurations my_kidz;
    CoreIDs the_mothers_of_my_children;
    if (message[1] > 0) {
        int child_support = message[1];
        the_mothers_of_my_children.reserve(child_support);
        MPI_Status status;
        MPI_Recv(
                the_mothers_of_my_children.data(),
                child_support,
                MPI_INT,
                0,
                0,
                MPI_COMM_WORLD,
                &status
                );

        reqs.reserve(child_support);
        int num_atoms = CORE_PROPAGATOR.atoms.size();
        my_kidz.resize(
                child_support,
                Coordinates (num_atoms, Point(3))
                );

        // set up the requests to get walkers from cores
        for ( int i = 0; i<child_support; i++ ) {
            MPI_Irecv(
                my_kidz[i].data(),
                num_atoms * 3,
                MPI_DOUBLE,
                the_mothers_of_my_children[i],
                0,
                MPI_COMM_WORLD,
                &reqs[i]
                );
        }

    }

    // if we're sending children set up a Send to get them




}

RawPotentialBuffer WalkerPropagator::get_pots() {

    auto pots = (RawPotentialBuffer) malloc(prop_steps*walkers.size());

    for (int j = 0; j < walkers.size(); j++) { // number of walkers we're holding onto
        for (int i = 0; i < prop_steps; i++) { // number of steps per walker
            // propagate the j-th walker forward
            for (int n = 0; n < atoms.size(); n++ ) {
                for (int m = 0; m < 3; m++ ) {
                    walkers[j][n][m] += distributions[m](engine);
                }
            }
            // call potential on newly moved walker and stick this in pots
            pots[j*prop_steps + i] = _doopAPot(walkers[j], atoms);
        }
    }

    return pots;

}