// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include <array>

#include <cstone/cuda/device_vector.h>

#include "scalar.h"

namespace chora
{

struct ParticleList
{
	cstone::DeviceVector<scalar> d_x;
	cstone::DeviceVector<scalar> d_y;
	cstone::DeviceVector<scalar> d_z;
	cstone::DeviceVector<scalar> d_vx;
	cstone::DeviceVector<scalar> d_vy;
	cstone::DeviceVector<scalar> d_vz;
	cstone::DeviceVector<scalar> d_ex;
	cstone::DeviceVector<scalar> d_ey;
	cstone::DeviceVector<scalar> d_ez;
	cstone::DeviceVector<scalar> d_h;
	cstone::DeviceVector<scalar> d_q;	// actual charge of the macroparticle, e.g., qe * np2c
	cstone::DeviceVector<scalar> d_m;	// actual mass of the macroparticle, e.g., me * np2c

        cstone::DeviceVector<unsigned> d_spid;

	void test();

	void add(cstone::DeviceVector<scalar>& d_x, cstone::DeviceVector<scalar>& d_y, cstone::DeviceVector<scalar>& d_z, scalar h, scalar q = 1, scalar m = 1, unsigned spid = 0);
	void add(cstone::DeviceVector<scalar>& d_x, cstone::DeviceVector<scalar>& d_y, cstone::DeviceVector<scalar>& d_z, cstone::DeviceVector<scalar>& d_vx, cstone::DeviceVector<scalar>& d_vy, cstone::DeviceVector<scalar>& d_vz, scalar h, scalar q = 1, scalar m = 1, unsigned spid = 0);
	uint64_t size();

	void correctField();

	std::array<scalar, 6> getBounds();
};

}
