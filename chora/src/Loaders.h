#pragma once

#include "ParticleList.cuh"
#include "Maxwellian.h"

namespace chora
{

class Loaders
{
public:
	static void loadMaxwellianSphere(chora::Maxwellian* f, double h, double q, double m, double np2c, int spid, chora::ParticleList* plist, unsigned npsphere, double radius);
};

}
