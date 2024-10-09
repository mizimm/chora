#pragma once

#include "scalar.h"
#include "ParticleList.cuh"
#include "Maxwellian.h"

namespace chora
{

class Loaders
{
public:
	static void loadMaxwellianSphere(chora::Maxwellian* f, scalar h, scalar q, scalar m, scalar np2c, int spid, chora::ParticleList* plist, unsigned npsphere, scalar radius);
};

}
