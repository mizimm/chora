// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

namespace chora
{

class Maxwellian
{
public:
	Maxwellian();
	Maxwellian(double vdx, double vdy, double vdz, double vth);

	double vdx;
	double vdy;
	double vdz;
	double vth;
};

}
