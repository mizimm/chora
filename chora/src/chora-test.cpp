/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *   2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief GTest MPI driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#include <array>
#include <chrono>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "scalar.h"
#include "ParticleList.cuh"
#include "ParticleAdvancer.cuh"
#include "Constants.h"
#include "Random.h"
#include "ErfInv.h"
#include "Maxwellian.h"
#include "Solvers.h"
#include "util.cuh"

namespace chora
{

std::chrono::time_point<std::chrono::high_resolution_clock> tic()
{
	return std::chrono::high_resolution_clock::now();
}

double toc(std::chrono::time_point<std::chrono::high_resolution_clock> starttime)	// get elapsed time in ms
{
        return std::chrono::duration_cast<std::chrono::milliseconds>(tic()-starttime).count();
}

std::string enumerateFilename(std::string prefix, int i, std::string extension)
{
	return prefix + "_" + std::to_string(i) + "." + extension;
}

void loadMaxwellianSphere(Maxwellian* f, scalar h, scalar q, scalar m, scalar np2c, int spid, chora::ParticleList* plist, unsigned npsphere, scalar radius)
{
	unsigned npbox = 6. / M_PI * (scalar)npsphere;

	std::array<scalar, 6> bounds{-radius, radius, -radius, radius, -radius, radius};

	thrust::host_vector<scalar> h_x;
	thrust::host_vector<scalar> h_y;
	thrust::host_vector<scalar> h_z;
	thrust::host_vector<scalar> h_vx;
	thrust::host_vector<scalar> h_vy;
	thrust::host_vector<scalar> h_vz;

	for (unsigned i = 0; i < npbox; i++)
	{
		scalar x = Random::between(-radius, radius);
		scalar y = Random::between(-radius, radius);
		scalar z = Random::between(-radius, radius);
		scalar r = sqrt(x*x + y*y + z*z);
		if (r > radius)
		{
			continue;
		}
		scalar vx = f->vdx + f->vth *  ErfInv::value(Random::pmone());
		scalar vy = f->vdy + f->vth *  ErfInv::value(Random::pmone());
		scalar vz = f->vdz + f->vth *  ErfInv::value(Random::pmone());
		//
		h_x.push_back(x);
		h_y.push_back(y);
		h_z.push_back(z);
		h_vx.push_back(vx);
		h_vy.push_back(vy);
		h_vz.push_back(vz);
	}

	thrust::device_vector<scalar> d_x = h_x;
	thrust::device_vector<scalar> d_y = h_y;
	thrust::device_vector<scalar> d_z = h_z;
	thrust::device_vector<scalar> d_vx = h_vx;
	thrust::device_vector<scalar> d_vy = h_vy;
	thrust::device_vector<scalar> d_vz = h_vz;
	plist->add(d_x, d_y, d_z, d_vx, d_vy, d_vz, h, q*np2c, m*np2c, spid);
}

void sphereTestElectronsProtons()
{
	// constants
	scalar dt = 1e-6;//5e-7;
	scalar n0 = 5e6;
	scalar np2c = 5e8;
	scalar radius = 50;
	scalar vdx = 0;
	scalar vdy = 0;
	scalar vdz = 0;
	scalar vthe = 2e5;
	scalar vthi = vthe * sqrt(Constants::ME / Constants::MP);

	// compute particle size
	scalar spherevol = 4 * M_PI * pow(radius, 3) / 3;
	unsigned np = n0 * spherevol / np2c;
	scalar h = cbrt(spherevol/np);

	// load particles
	ParticleList plist;
	Maxwellian fe(vdx, vdy, vdz, vthe);
	Maxwellian fi(vdx, vdy, vdz, vthi);
	loadMaxwellianSphere(&fe, h, -Constants::QE, Constants::ME, np2c, 0, &plist, np, radius);
	loadMaxwellianSphere(&fi, h, +Constants::QE, Constants::MP, np2c, 1, &plist, np, radius);

	// run and save timesteps
	int maxstep = 100;
	int dmpstride = 1;
	for (int i = 0; i <= maxstep; i ++)
	{
		std::cout << "Step " << i << std::endl;

		// solve
		Solvers::multipoleGpu(&plist);
//		Solvers::directGpu(&plist);
//		Solvers::directCpu(&plist, 2*h);

		// write
		if (i==0 || !(i%dmpstride))
		{
			writeParticlesBin(&plist, enumerateFilename("out", i, "bin"));
		}

		// advance
		ParticleAdvancer(dt).apply(&plist, dt);
	}
}


/*void pairTest()
{
	chora::ParticleList plist;
	chora::PairLoader::load(&plist, {-0.5, 0, 0}, {0.5, 0, 0}, 0.01, 0.01, -1, 1);
	solveDirect(&plist, {-1, 1, -1, 1, -1, 1});
	chora::writeParticlesBin(&plist, "out.bin");
}*/

}

int main(int argc, char** argv)
{
	MPI_Init(NULL, NULL);
	int rank = 0, numRanks = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

	chora::sphereTestElectronsProtons();

	MPI_Finalize();
}
