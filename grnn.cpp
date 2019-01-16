
#include "grnn.h"


GRNN::GRNN(char * X, char * Y, int row, int col)
{
    load(X, X_obs, row, col);
    load(Y, Y_obs, row, 1);

    inp_row = row;
    inp_col = col;

    mean.resize(inp_col);
    stdev.resize(inp_col);

    stand(X_obs, mean, stdev);
}


void GRNN::load(char * file, std::vector < std::vector <double> > & input, int row, int col)
{
    std::ifstream inp;
    inp.open(file);

    input.resize(row);
    for(int i = 0; i < row; i++)
        input[i].resize(col);

    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            inp >> input[i][j];
}


void GRNN::load_mod(char * file, int row, int col)
{
    load(file, X_mod, row, col);

    apply_stand(X_mod, mean, stdev);

    mod_row = row;

    Y_mod.resize(mod_row);
}


void GRNN::model()                                                                  // ?mtx form
{
    std::cout << "SIGMA :   " << sigma << std::endl;

    std::vector <double> W(inp_row);
    double D = 0;
    double num = 0;                                                                 // numerator
    double den = 0;                                                                 // denominator

    // loop over k modelled inps
    for(int k = 0; k < mod_row; k++)
    {
        // loop over i inp rows
        for(int i = 0; i < inp_row; i++)
        {
            D = 0;

            // loop over j inp cols
            for(int j = 0; j < inp_col; j++)
                D += (X_obs[i][j] - X_mod[k][j]) * (X_obs[i][j] - X_mod[k][j]);     // D = (X - Xi)^2

            W[i] = exp(-D / (2 * sigma * sigma));                                   // W = exp(-D/2sig^2)
        }

        num = 0;
        den = 0;

        for(int i = 0; i < inp_row; i++)
        {
            num += W[i] * Y_obs[i][0];
            den += W[i];
        }

        Y_mod[k] = num / den;
    }
}


/*  Holdout method of finding best SIGMA
 *
 *  removes one sample at a time and
 *  estimates its Y[mod] value, then
 *  evaluates the mse of the entire
 *  sample dataset for each SIGMA in
 *  the specified range
 */
void GRNN::holdout(double low, double high, double inc)
{
    double current_sigma = low;
    std::vector <double> W(inp_row);

    double best_mse = 1e8;
    double mse = 0;

    double D = 0;
    double num = 0;                                                                     // numerator
    double den = 0;                                                                     // denominator

    while(current_sigma < high)
    {
        mse = 0;

        // k = removed sample
        for(int k = 0; k < inp_row; k++)
        {
            // i = row
            for(int i = 0; i < inp_row; i++)
            {
                // skip removed sample
                if(i == k)
                    continue;

                D = 0;

                // j = col
                for(int j = 0; j < inp_col; j++)
                    D += (X_obs[i][j] - X_obs[k][j]) * (X_obs[i][j] - X_obs[k][j]);     // D = (X - Xi)^2

                W[i] = exp(-D / (2 * current_sigma * current_sigma));                   // W = exp(-D/2sig^2)
            }

            num = 0;
            den = 0;

            for(int i = 0; i < inp_row; i++)
            {
                num += W[i] * Y_obs[i][0];
                den += W[i];
            }

            mse += ((num / den) - Y_obs[k][0])*((num / den) - Y_obs[k][0]);
        }

        mse /= (inp_row - 1);
        std::cout << "c_sigma: " << current_sigma << " mse: " << mse << std::endl;

        // update if mse[cur_sig] < mse[sig]
        if(mse < best_mse)
        {
            best_mse = mse;
            sigma = current_sigma;
            std::cout << "best sig:     " << sigma << std::endl;
        }

        current_sigma += inc;
    }
}

/*  Set SIGMA manually
 *
 */
void GRNN::set_sigma(double sig)
{
    sigma = sig;
}


void GRNN::print(std::ostream &out)
{
    for(int i = 0; i < mod_row; i++)
        out << Y_mod[i] << std::endl;
}


void GRNN::print(char * file)
{
    std::ofstream out;
    out.open(file);

    for(int i = 0; i < mod_row; i++)
        out << Y_mod[i] << std::endl;
}


/*  Standardization of the input vars
 *
 */
void GRNN::stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev)
{
    // MEAN
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            mean[j] += input[i][j];
        }
    }
    for(int i = 0; i < input[0].size(); i++)
        mean[i] /= input.size();

    // STDEV
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            stdev[j] += (input[i][j] - mean[j])*(input[i][j] - mean[j]);
        }
    }
    for(int i = 0; i < input[0].size(); i++)
        stdev[i] = sqrt(stdev[i] / input.size());

    // -> STAND
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            input[i][j] = input[i][j] / stdev[j];
        }
    }
}


void GRNN::apply_stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev)
{
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            input[i][j] = input[i][j] / stdev[j];
        }
    }
}
