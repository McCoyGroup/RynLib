
// The return type can be changed up if it's possible
// to directly get the coordinates at the C++ level from entos
// otherwise we can pull them of stdout on the python side

extern "C" {

double MillerGroup_entosPotential
    (std::vector< std::vector<double> >, std::vector<std::string>, bool hf_only = false);

}

