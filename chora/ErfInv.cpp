// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include "ErfInv.h"

#include <cmath>

namespace chora
{

const double ErfInv::ERFINVCONST=8./3./M_PI*(M_PI-3.)/(4.-M_PI);

double ErfInv::value(const double& x)	// from approximants paper
{
        double sqfac=2./M_PI/ERFINVCONST+log(1.-x*x)/2.;
        return (x<0?-1.:1.)*sqrt(-2./M_PI/ERFINVCONST-log(1-x*x)/2.+sqrt(sqfac*sqfac-1./ERFINVCONST*log(1.-x*x)));
}

}
