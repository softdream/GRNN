
#include "grnn.h"


int main()
{
    // constructor + load obs data
    GRNN net("data/X_obs.txt", "data/Y_obs.txt", 2265, 11);                     // X_obs, Y_obs, row, col


    // load mod data
    net.load_mod("data/X_mod.txt", 1510, 11);                                   // X_mod, row, col


    // find/set SIGMA
    net.holdout(0.1, 1, 0.1);                                                   // low, high, step
    //net.set_sigma(0.56);


    //
    net.model();


    net.print(std::cout);
    net.print("data/Y_mod.txt");




    return 0;
}
