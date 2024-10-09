// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include "ParticleList.cuh"
#include "Constants.h"

namespace chora
{

uint64_t ParticleList::size()
{
	return d_x.size();
}

void ParticleList::add(thrust::device_vector<double>& d_x, thrust::device_vector<double>& d_y, thrust::device_vector<double>& d_z, thrust::device_vector<double>& d_vx, thrust::device_vector<double>& d_vy, thrust::device_vector<double>& d_vz, double h, double q, double m, unsigned spid)
{
	this->d_x.insert(this->d_x.end(), d_x.begin(), d_x.end());
	this->d_y.insert(this->d_y.end(), d_y.begin(), d_y.end());
	this->d_z.insert(this->d_z.end(), d_z.begin(), d_z.end());

	this->d_vx.insert(this->d_vx.end(), d_vx.begin(), d_vx.end());
	this->d_vy.insert(this->d_vy.end(), d_vy.begin(), d_vy.end());
	this->d_vz.insert(this->d_vz.end(), d_vz.begin(), d_vz.end());

	int np = d_x.size();

	thrust::constant_iterator<double> it_zero(0);
	this->d_ex.insert(this->d_ex.end(), it_zero, it_zero + np);
	this->d_ey.insert(this->d_ey.end(), it_zero, it_zero + np);
	this->d_ez.insert(this->d_ez.end(), it_zero, it_zero + np);

	thrust::constant_iterator<double> it_q(q);
	thrust::constant_iterator<double> it_m(m);
	thrust::constant_iterator<double> it_h(h);
	this->d_q.insert(this->d_q.end(), it_q, it_q + np);
	this->d_m.insert(this->d_m.end(), it_m, it_m + np);
	this->d_h.insert(this->d_h.end(), it_h, it_h + np);

	thrust::constant_iterator<unsigned> it_spid(spid);
	this->d_spid.insert(this->d_spid.end(), it_spid, it_spid + np);
}

void ParticleList::add(thrust::device_vector<double>& d_x, thrust::device_vector<double>& d_y, thrust::device_vector<double>& d_z, double h, double q, double m, unsigned spid)
{
	int np = d_x.size();
	thrust::device_vector<double> d_vx(np);
	thrust::device_vector<double> d_vy(np);
	thrust::device_vector<double> d_vz(np);
	add(d_x, d_y, d_z, d_vx, d_vy, d_vz, h, q, m, spid);
}

void ParticleList::correctRyoanjiField()
{
      thrust::constant_iterator<double> it_coulomb(-Constants::COULOMB);
      thrust::transform(d_ex.begin(), d_ex.end(), it_coulomb, d_ex.begin(), thrust::multiplies<double>());
      thrust::transform(d_ey.begin(), d_ey.end(), it_coulomb, d_ey.begin(), thrust::multiplies<double>());
      thrust::transform(d_ez.begin(), d_ez.end(), it_coulomb, d_ez.begin(), thrust::multiplies<double>());
}

/*void ParticleList::test()
{
	thrust::host_vector<double> h_x = d_x;
	thrust::host_vector<double> h_y = d_y;
	thrust::host_vector<double> h_z = d_z;

	printf("%f %f %f\n", h_x[100], h_y[100], h_z[100]);
}*/

std::array<double, 6> ParticleList::getBounds()
{
	if (size() == 0)
	{
		return {0, 0, 0, 0, 0, 0};
	}

	double xmin = *thrust::min_element(d_x.begin(), d_x.end());
	double xmax = *thrust::max_element(d_x.begin(), d_x.end());
	double ymin = *thrust::min_element(d_y.begin(), d_y.end());
	double ymax = *thrust::max_element(d_y.begin(), d_y.end());
	double zmin = *thrust::min_element(d_z.begin(), d_z.end());
	double zmax = *thrust::max_element(d_z.begin(), d_z.end());

	return {xmin, xmax, ymin, ymax, zmin, zmax};
}

void ParticleList::initRyoanjiIds(thrust::device_vector<unsigned>& d_tmpids)
{
	d_tmpids.resize(size());
	thrust::sequence(d_tmpids.begin(), d_tmpids.end());
}

void ParticleList::descrambleRyoanjiFieldComponents(thrust::device_vector<unsigned>& d_tmpids, thrust::device_vector<double>& d_ex, thrust::device_vector<double>& d_ey, thrust::device_vector<double>& d_ez)
{
	// TODO: deal with dryoanji domain.startIndex() offset
	thrust::scatter(d_ex.begin(), d_ex.end(), d_tmpids.begin(), this->d_ex.begin());
	thrust::scatter(d_ey.begin(), d_ey.end(), d_tmpids.begin(), this->d_ey.begin());
	thrust::scatter(d_ez.begin(), d_ez.end(), d_tmpids.begin(), this->d_ez.begin());
}

//    const auto* map  = d_id.data() + domain.startIndex();
//    size_t numElements = domain.nParticles();
//        chora::scatterGpu(map, numElements, d_ex.data() + domain.startIndex(), plist->d_ex.data() + domain.startIndex());
//    chora::scatterGpu(map, numElements, d_ey.data() + domain.startIndex(), plist->d_ey.data() + domain.startIndex());
//    chora::scatterGpu(map, numElements, d_ez.data() + domain.startIndex(), plist->d_ez.data() + domain.startIndex());
}
