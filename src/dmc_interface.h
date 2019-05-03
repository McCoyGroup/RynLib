#ifndef ENTOS_ML_DMC_INTERFACE_H
#define ENTOS_ML_DMC_INTERFACE_H

#include <vector>
#include <string>

double MillerGroup_entosPotential
        (const std::vector< std::vector<double> > , const std::vector<std::string>, bool hf_only = false);

#endif //ENTOS_ML_DMC_INTERFACE_H
