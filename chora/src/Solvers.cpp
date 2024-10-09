// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC
// © 2024 Sebastian Keller

#include "Solvers.h"

#include <array>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#include <thrust/transform.h>
//#include <thrust/tuple.h>
//#include <thrust/iterator/constant_iterator.h>
//#include <thrust/sequence.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/domain/domain.hpp"
#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"

#include "keytype.h"
#include "scalar.h"
#include "Constants.h"

namespace chora
{

void Solvers::multipoleGpu(chora::ParticleList* plist)//, scalar G = 1.0, scalar theta = 0.5)
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

	using acctype = cstone::GpuTag;
	cstone::Domain<keytype, scalar, acctype> domain(
		thisRank,
		numRanks,
		bucketSizeGlobal,
		bucketSizeFocus,
		theta,
		box
	);

	thrust::device_vector<keytype> d_keys = std::vector<keytype>(numParticles);
	thrust::device_vector<scalar>   s1, s2, s3; // scratch buffers for sorting, reordering, etc

	thrust::device_vector<scalar> d_x = plist->d_x;
	thrust::device_vector<scalar> d_y = plist->d_y;
	thrust::device_vector<scalar> d_z = plist->d_z;
	thrust::device_vector<scalar> d_h = plist->d_h;
	thrust::device_vector<scalar> d_q = plist->d_q;

	thrust::device_vector<unsigned> d_tmpids;
	plist->initRyoanjiIds(d_tmpids);	// TODO: this should not be a ParticleList method; d_tmpids should have a length of domain.nParticlesWithHalos()

	//! Build octrees, decompose domain and distribute particles and halos, may resize buffers
	domain.syncGrav(
		d_keys,
		d_x, d_y, d_z, d_h, d_q,
		std::tie(d_tmpids),
		std::tie(s1, s2, s3)
	);

	//! halos for x,y,z,h are already exchanged in syncGrav
	domain.exchangeHalos(
		std::tie(d_q),
		s1, s2
	);

	thrust::device_vector<scalar> d_ex = std::vector<scalar>(domain.nParticlesWithHalos());
	thrust::device_vector<scalar> d_ey = std::vector<scalar>(domain.nParticlesWithHalos());
	thrust::device_vector<scalar> d_ez = std::vector<scalar>(domain.nParticlesWithHalos());

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

//	std::cout << domain.startIndex() << " " << domain.endIndex() <<std::endl;

//	// copy particle accelerations into plist
//	thrust::copy(d_ex.begin() + domain.startIndex(), d_ex.begin() + domain.endIndex(), plist->d_ex.begin());
//	thrust::copy(d_ey.begin() + domain.startIndex(), d_ey.begin() + domain.endIndex(), plist->d_ey.begin());
//	thrust::copy(d_ez.begin() + domain.startIndex(), d_ez.begin() + domain.endIndex(), plist->d_ez.begin());

	plist->descrambleRyoanjiFieldComponents(d_tmpids, d_ex, d_ey, d_ez);
	plist->correctRyoanjiField();
}

void Solvers::directGpu(chora::ParticleList* plist)
{
	std::array<scalar, 6> bounds = plist->getBounds();
	cstone::Box<scalar> box{bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]};

	thrust::device_vector<scalar> d_x = plist->d_x;
	thrust::device_vector<scalar> d_y = plist->d_y;
	thrust::device_vector<scalar> d_z = plist->d_z;
	thrust::device_vector<scalar> d_h = plist->d_h;
	thrust::device_vector<scalar> d_q = plist->d_q;

	int numBodies = plist->size();
	int numShells = 0;

	thrust::device_vector<scalar> d_ex = std::vector<scalar>(numBodies);
	thrust::device_vector<scalar> d_ey = std::vector<scalar>(numBodies);
	thrust::device_vector<scalar> d_ez = std::vector<scalar>(numBodies);
	thrust::device_vector<scalar> d_phi = std::vector<scalar>(numBodies);

	ryoanji::directSum(0, numBodies, numBodies, {box.lx(), box.ly(), box.lz()}, numShells, rawPtr(d_x), rawPtr(d_y),
              rawPtr(d_z), rawPtr(d_q), rawPtr(d_h), rawPtr(d_phi), rawPtr(d_ex), rawPtr(d_ey), rawPtr(d_ez));

	// copy particle accelerations into plist
	thrust::copy(d_ex.begin(), d_ex.end(), plist->d_ex.begin());
	thrust::copy(d_ey.begin(), d_ey.end(), plist->d_ey.begin());
	thrust::copy(d_ez.begin(), d_ez.end(), plist->d_ez.begin());

	plist->correctRyoanjiField();
}

void Solvers::directCpu(chora::ParticleList* plist, scalar dmin)
{
	thrust::host_vector<scalar> h_x = plist->d_x;
	thrust::host_vector<scalar> h_y = plist->d_y;
	thrust::host_vector<scalar> h_z = plist->d_z;
	thrust::host_vector<scalar> h_q = plist->d_q;

	thrust::host_vector<scalar> h_ex(plist->size());
	thrust::host_vector<scalar> h_ey(plist->size());
	thrust::host_vector<scalar> h_ez(plist->size());

	thrust::fill(h_ex.begin(), h_ex.end(), 0);
	thrust::fill(h_ey.begin(), h_ey.end(), 0);
	thrust::fill(h_ez.begin(), h_ez.end(), 0);

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



}
