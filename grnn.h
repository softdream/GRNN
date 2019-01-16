
#ifndef GRNN_GRNN_H
#define GRNN_GRNN_H


#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>


class GRNN
{
public:
    GRNN(char * X, char * Y, int row, int col);

    void load(char * file, std::vector < std::vector < double > > & input, int row, int col);
    void load_mod(char * file, int row, int col);

    void set_sigma(double sig);

    void model();
    void holdout(double low, double high, double inc);

    void print(std::ostream &out);
    void print(char * file);

    void stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev);
    void apply_stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev);


private:
    std::vector < std::vector < double > > X_obs;
    std::vector < std::vector < double > > Y_obs;

    std::vector < std::vector < double > > X_mod;
    std::vector < double > Y_mod;

    std::vector < double > mean;
    std::vector < double > stdev;

    unsigned int inp_row;
    unsigned int inp_col;
    unsigned int mod_row;

    double sigma;
};



#endif //GRNN_GRNN_H
