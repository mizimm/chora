// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include <thrust/tuple.h>
#include "scalar.h"
#include "ParticleList.cuh"

namespace chora
{

class ParticleAdvancer
{
protected:
	scalar dt;

public:

	ParticleAdvancer(scalar dt);

	__host__ __device__ thrust::tuple<scalar,scalar,scalar,scalar,scalar,scalar> operator() (thrust::tuple<scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar> t);

	void apply(ParticleList* plist, scalar dt);
};

}
