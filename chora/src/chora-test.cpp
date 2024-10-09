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

#include <chrono>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ParticleList.cuh"
#include "ParticleAdvancer.cuh"
#include "Constants.h"
#include "Random.h"
#include "ErfInv.h"
#include "Maxwellian.h"
#include "Solvers.h"
#include "Loaders.h"
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

/*void createSideInjectors(vector<QuadEmitter*>& injectors, const Box& bbox, const vector<BoxSide>& sides, Maxwellian* f, double n0, double np2c, double dnml)
{
        for (BoxSide side : sides)
        {
                Quad* quad = GeomUtil::createSide(bbox, side, BoxNormals::INWARD);
                OrthonormalBasis* basis = new OrthonormalBasis(quad->getNormal());
                FluxSampler* fluxsamp = new MaxwellianFluxSampler(f, basis);
                double ptclrate = n0 * fluxsamp->getMeanNormalVelocity() * quad->getArea() / np2c;
                QuadEmitter* emitter = new QuadEmitter(quad, fluxsamp, ptclrate, dnml);
                injectors.push_back(emitter);
        }
}*/

void sphereTestElectronsProtons()
{
	// constants
	double dt = 1e-6;//5e-7;
	double n0 = 5e6;
	double np2c = 5e8;
	double radius = 50;
	double vdx = 0;
	double vdy = 0;
	double vdz = 0;
	double vthe = 2e5;
	double vthi = vthe * sqrt(Constants::ME / Constants::MP);

	// compute particle size
	double spherevol = 4 * M_PI * pow(radius, 3) / 3;
	unsigned np = n0 * spherevol / np2c;
	double h = cbrt(spherevol/np);

	// load particles
	ParticleList plist;
	Maxwellian fe(vdx, vdy, vdz, vthe);
	Maxwellian fi(vdx, vdy, vdz, vthi);
	Loaders::loadMaxwellianSphere(&fe, h, -Constants::QE, Constants::ME, np2c, 0, &plist, np, radius);
	Loaders::loadMaxwellianSphere(&fi, h, +Constants::QE, Constants::MP, np2c, 1, &plist, np, radius);

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
