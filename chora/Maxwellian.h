// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include "scalar.h"

namespace chora
{

class Maxwellian
{
public:
	Maxwellian();
	Maxwellian(scalar vdx, scalar vdy, scalar vdz, scalar vth);

	scalar vdx;
	scalar vdy;
	scalar vdz;
	scalar vth;
};

}
