// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include <array>

#include <thrust/device_vector.h>

namespace chora
{

struct ParticleList
{
	thrust::device_vector<double> d_x;
	thrust::device_vector<double> d_y;
	thrust::device_vector<double> d_z;
	thrust::device_vector<double> d_vx;
	thrust::device_vector<double> d_vy;
	thrust::device_vector<double> d_vz;
	thrust::device_vector<double> d_ex;
	thrust::device_vector<double> d_ey;
	thrust::device_vector<double> d_ez;
	thrust::device_vector<double> d_h;
	thrust::device_vector<double> d_q;	// actual charge of the macroparticle, e.g., qe * np2c
	thrust::device_vector<double> d_m;	// actual mass of the macroparticle, e.g., me * np2c

	thrust::device_vector<unsigned> d_spid;

	void add(thrust::device_vector<double>& d_x, thrust::device_vector<double>& d_y, thrust::device_vector<double>& d_z, double h, double q = 1, double m = 1, unsigned spid = 0);
	void add(thrust::device_vector<double>& d_x, thrust::device_vector<double>& d_y, thrust::device_vector<double>& d_z, thrust::device_vector<double>& d_vx, thrust::device_vector<double>& d_vy, thrust::device_vector<double>& d_vz, double h, double q = 1, double m = 1, unsigned spid = 0);
	uint64_t size();

	// ryoanji-specific methods
	void correctRyoanjiField();
	void initRyoanjiIds(thrust::device_vector<unsigned>& d_tmpids);
	void descrambleRyoanjiFieldComponents(thrust::device_vector<unsigned>& d_tmpids, thrust::device_vector<double>& d_ex, thrust::device_vector<double>& d_ey, thrust::device_vector<double>& d_ez);

	std::array<double, 6> getBounds();
};

}
