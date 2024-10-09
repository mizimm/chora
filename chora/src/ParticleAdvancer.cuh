// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include <thrust/tuple.h>
#include "ParticleList.cuh"

namespace chora
{

class ParticleAdvancer
{
protected:
	double dt;

public:

	ParticleAdvancer(double dt);

	__host__ __device__ thrust::tuple<double,double,double,double,double,double> operator() (thrust::tuple<double,double,double,double,double,double,double,double,double,double,double> t);

	void apply(ParticleList* plist, double dt);
};

}
