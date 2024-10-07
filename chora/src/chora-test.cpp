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

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
//#include "coord_samples/random.hpp"

#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"

#include "ParticleList.cuh"
#include "ParticleAdvancer.cuh"
#include "Constants.h"
#include "Random.h"
#include "ErfInv.h"
#include "Maxwellian.h"
#include "util.cuh"

#include "scalar.h"
#include "keytype.h"

/*void solveDirect(chora::ParticleList* plist, std::array<scalar, 6> bounds)
{
	cstone::Vec3<scalar> boxsize{bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]};

	thrust::device_vector<scalar> d_phi;
	ryoanji::directSum(
		0,
		plist->size(),
		plist->size(),
		boxsize,
		0,
		rawPtr(plist->d_x),
		rawPtr(plist->d_y),
		rawPtr(plist->d_z),
		rawPtr(plist->d_q),
		rawPtr(plist->d_h),
		rawPtr(d_phi),
		rawPtr(plist->d_ax),
		rawPtr(plist->d_ay),
		rawPtr(plist->d_az)
	);
}*/

void solveMultipoleGpu(chora::ParticleList* plist)//, scalar G = 1.0, scalar theta = 0.5)
{
	std::array<scalar, 6> bounds = plist->getBounds();

	// explicitly run without MPI parallelism
	int thisRank = 0;
	int numRanks = 1;

	size_t numParticles = plist->size();//numParticlesGlobal / numRanks;
	unsigned bucketSizeFocus = 64;
	unsigned numGlobalNodesPerRank = 100;
	unsigned bucketSizeGlobal = std::max(size_t(bucketSizeFocus), plist->size()/ (numGlobalNodesPerRank * numRanks));
	scalar G = 1.0;
	float theta = 0;	// use low theta to effectively get direct solve
	cstone::Box<scalar> box{bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]};

	// TODO: ask Sebastian if the following comment means that h is a particle neighbor size, and not the particle size itself
	// radius that includes ~30 neighbor particles on average
	//double hmean = std::cbrt(30.0 / 0.523 / double(numParticlesGlobal) * box.lx() * box.ly() * box.lz());

	using acctype = cstone::GpuTag;
	cstone::Domain<keytype, scalar, acctype> domain(
		thisRank,
		numRanks,
		bucketSizeGlobal,
		bucketSizeFocus,
		theta,
		box
	);

	cstone::DeviceVector<keytype> d_keys = std::vector<keytype>(numParticles);
	cstone::DeviceVector<scalar>   s1, s2, s3; // scratch buffers for sorting, reordering, etc

	cstone::DeviceVector<scalar> d_x = plist->d_x;
	cstone::DeviceVector<scalar> d_y = plist->d_y;
	cstone::DeviceVector<scalar> d_z = plist->d_z;
	cstone::DeviceVector<scalar> d_h = plist->d_h;
	cstone::DeviceVector<scalar> d_q = plist->d_q;

	//! Build octrees, decompose domain and distribute particles and halos, may resize buffers
	domain.syncGrav(
		d_keys,
		d_x,
		d_y,
		d_z,
		d_h,
		d_q,
		std::tuple{}, std::tie(s1, s2, s3)
	);

	//! halos for x,y,z,h are already exchanged in syncGrav
	domain.exchangeHalos(
		std::tie(d_q), s1, s2
	);

	cstone::DeviceVector<scalar> d_ex = std::vector<scalar>(domain.nParticlesWithHalos());
	cstone::DeviceVector<scalar> d_ey = std::vector<scalar>(domain.nParticlesWithHalos());
	cstone::DeviceVector<scalar> d_ez = std::vector<scalar>(domain.nParticlesWithHalos());

	//! includes tree plus associated information, like nearby ranks, assignment, counts, MAC spheres, etc
	const cstone::FocusedOctree<keytype, scalar, acctype>& focusTree = domain.focusTree();
	//! the focused octree on GPU, structure only
	auto octree = focusTree.octreeViewAcc();

	constexpr int P = 4;
	using multipoletype = ryoanji::SphericalMultipole<scalar, P>;
	ryoanji::MultipoleHolder<scalar, scalar, scalar, scalar, scalar, keytype, multipoletype> multipoleHolder;

	std::vector<multipoletype> multipoles(octree.numNodes);
	auto grp = multipoleHolder.computeSpatialGroups(
		domain.startIndex(),
		domain.endIndex(),
		rawPtr(d_x),
		rawPtr(d_y),
		rawPtr(d_z),
		rawPtr(d_h),
//		rawPtr(plist->d_x),
//		rawPtr(plist->d_y),
//		rawPtr(plist->d_z),
//		rawPtr(plist->d_h),
		domain.focusTree(),
		domain.layout().data(),
		domain.box()
	);
	multipoleHolder.upsweep(
		rawPtr(d_x),
		rawPtr(d_y),
		rawPtr(d_z),
		rawPtr(d_q),
//		rawPtr(plist->d_x),
//		rawPtr(plist->d_y),
//		rawPtr(plist->d_z),
//		rawPtr(plist->d_q),
		domain.globalTree(),
		domain.focusTree(),
		domain.layout().data(),
		multipoles.data()
	);

	// compute accelerations for locally owned particles based on globally valid multipoles and halo particles are in [0:domain.startIndex()] and in [domain.endIndex():domain.nParticlesWithHalos()]
	multipoleHolder.compute(
		grp,
		rawPtr(d_x),
		rawPtr(d_y),
		rawPtr(d_z),
		rawPtr(d_q),
		rawPtr(d_h),
//		rawPtr(plist->d_x),
//		rawPtr(plist->d_y),
//		rawPtr(plist->d_z),
//		rawPtr(plist->d_q),
//		rawPtr(plist->d_h),
		G,
		0,
		box,
		rawPtr(d_ex),
		rawPtr(d_ey),
		rawPtr(d_ez)
	);

//	std::cout << d_ax.size() << " " << plist->d_ax.size() << " " << d_ax.startIndex() << " " << d_ax.endIndex() << std::endl;

	// copy particle accelerations into plist
	memcpyD2D(d_ex.data() + domain.startIndex(), domain.endIndex() - domain.startIndex(), plist->d_ex.data());
	memcpyD2D(d_ey.data() + domain.startIndex(), domain.endIndex() - domain.startIndex(), plist->d_ey.data());
	memcpyD2D(d_ez.data() + domain.startIndex(), domain.endIndex() - domain.startIndex(), plist->d_ez.data());

	plist->correctField();
}

void solveDirectGpu(chora::ParticleList* plist)
{
	std::array<scalar, 6> bounds = plist->getBounds();
	cstone::Box<scalar> box{bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]};

	cstone::DeviceVector<scalar> d_x = plist->d_x;
	cstone::DeviceVector<scalar> d_y = plist->d_y;
	cstone::DeviceVector<scalar> d_z = plist->d_z;
	cstone::DeviceVector<scalar> d_h = plist->d_h;
	cstone::DeviceVector<scalar> d_q = plist->d_q;

	int numBodies = plist->size();
	int numShells = 0;

	cstone::DeviceVector<scalar> d_ex = std::vector<scalar>(numBodies);
	cstone::DeviceVector<scalar> d_ey = std::vector<scalar>(numBodies);
	cstone::DeviceVector<scalar> d_ez = std::vector<scalar>(numBodies);
	cstone::DeviceVector<scalar> d_phi = std::vector<scalar>(numBodies);

	ryoanji::directSum(0, numBodies, numBodies, {box.lx(), box.ly(), box.lz()}, numShells, rawPtr(d_x), rawPtr(d_y),
              rawPtr(d_z), rawPtr(d_q), rawPtr(d_h), rawPtr(d_phi), rawPtr(d_ex), rawPtr(d_ey), rawPtr(d_ez));

	// copy particle accelerations into plist
	memcpyD2D(d_ex.data(), d_ex.size(), plist->d_ex.data());
	memcpyD2D(d_ey.data(), d_ey.size(), plist->d_ey.data());
	memcpyD2D(d_ez.data(), d_ez.size(), plist->d_ez.data());

	plist->correctField();
}

// chora namespace contents are Copyright Mike Zimmerman JHUAPL 2024

#include "omp.h"

namespace chora
{

void solveDirectCpu(chora::ParticleList* plist, scalar dmin)
{
	std::vector<scalar> h_x = toHost(plist->d_x);
	std::vector<scalar> h_y = toHost(plist->d_y);
	std::vector<scalar> h_z = toHost(plist->d_z);
	std::vector<scalar> h_q = toHost(plist->d_q);

	std::vector<scalar> h_ex(plist->size());
	std::vector<scalar> h_ey(plist->size());
	std::vector<scalar> h_ez(plist->size());

	std::fill(h_ex.begin(), h_ex.end(), 0);
	std::fill(h_ey.begin(), h_ey.end(), 0);
	std::fill(h_ez.begin(), h_ez.end(), 0);

//	int cnt = 0;

	#pragma omp parallel for shared(plist, h_x, h_y, h_z, h_q, h_ex, h_ey, h_ez)
	for (int i = 0; i < plist->size(); i++)	// target
	{
//		if (!(i%1000))
//		{
//			#pragma omp critical
//			std::cout << cnt << "/" << plist->size() << std::endl;
//			#pragma omp atomic
//			cnt+=1000;
//		}
		for (int j = 0; j < plist->size(); j++)	// source
		{
			scalar dx = h_x[i] - h_x[j];
			scalar dy = h_y[i] - h_y[j];
			scalar dz = h_z[i] - h_z[j];
			scalar dr = sqrt(dx*dx + dy*dy + dz*dz);
			dr = fmax(dr, dmin);
			if (dr == 0)
			{
				continue;
			}
			scalar dr3 = dr*dr*dr;
			scalar ex = h_q[j] * Constants::COULOMB  * dx / dr3;
			scalar ey = h_q[j] * Constants::COULOMB  * dy / dr3;
			scalar ez = h_q[j] * Constants::COULOMB  * dz / dr3;
			h_ex[i] += ex;
			h_ey[i] += ey;
			h_ez[i] += ez;
		}
	}

	plist->d_ex = h_ex;
	plist->d_ey = h_ey;
	plist->d_ez = h_ez;
}

std::string enumerateFilename(std::string prefix, int i, std::string extension)
{
	return prefix + "_" + std::to_string(i) + "." + extension;
}

void loadMaxwellianSphere(Maxwellian* f, scalar h, scalar q, scalar m, scalar np2c, int spid, chora::ParticleList* plist, unsigned npsphere, scalar radius)
{
	unsigned npbox = 6. / M_PI * (scalar)npsphere;

	std::array<scalar, 6> bounds{-radius, radius, -radius, radius, -radius, radius};

	std::vector<scalar> h_x;
	std::vector<scalar> h_y;
	std::vector<scalar> h_z;
	std::vector<scalar> h_vx;
	std::vector<scalar> h_vy;
	std::vector<scalar> h_vz;

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

	cstone::DeviceVector<scalar> d_x = h_x;
	cstone::DeviceVector<scalar> d_y = h_y;
	cstone::DeviceVector<scalar> d_z = h_z;
	cstone::DeviceVector<scalar> d_vx = h_vx;
	cstone::DeviceVector<scalar> d_vy = h_vy;
	cstone::DeviceVector<scalar> d_vz = h_vz;
	plist->add(d_x, d_y, d_z, d_vx, d_vy, d_vz, h, -q*np2c, m*np2c, spid);
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
	int maxstep = 1000;
	int dmpstride = 1;
	for (int i = 0; i <= maxstep; i ++)
	{
		std::cout << "Step " << i << std::endl;

		// solve
		//solveMultipoleGpu(&plist);
		//solveDirectGpu(&plist);
		solveDirectCpu(&plist, h);

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
