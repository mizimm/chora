// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"

#include "ParticleList.cuh"
#include "Constants.h"

namespace chora
{

uint64_t ParticleList::size()
{
	return d_x.size();
}

void append(cstone::DeviceVector<scalar>& dst, const cstone::DeviceVector<scalar>& src)
{
    auto origSize = dst.size();
    dst.resize(origSize + src.size());
    memcpyD2D(src.data(), src.size(), dst.data() + origSize);
}

template<class T>
void append(cstone::DeviceVector<T>& dst, size_t numElements, T x)
{
    auto origSize = dst.size();
    dst.resize(origSize + numElements);
    thrust::fill(thrust::device, dst.data() + origSize, dst.data() + dst.size(), x);
}

void ParticleList::add(cstone::DeviceVector<scalar>& d_x, cstone::DeviceVector<scalar>& d_y,
                       cstone::DeviceVector<scalar>& d_z, cstone::DeviceVector<scalar>& d_vx,
                       cstone::DeviceVector<scalar>& d_vy, cstone::DeviceVector<scalar>& d_vz, scalar h, scalar q,
                       scalar m, unsigned spid)
{
    unsigned numParticlesBefore = this->size();

    append(this->d_x, d_x);
    append(this->d_y, d_y);
	append(this->d_z, d_z);

	append(this->d_vx, d_vx);
	append(this->d_vy, d_vy);
	append(this->d_vz, d_vz);

    auto numParticlesToAdd = d_x.size();

    append(this->d_ex, numParticlesToAdd, scalar(0));
    append(this->d_ey, numParticlesToAdd, scalar(0));
    append(this->d_ez, numParticlesToAdd, scalar(0));

    append(this->d_q, numParticlesToAdd, q);
    append(this->d_m, numParticlesToAdd, m);
    append(this->d_h, numParticlesToAdd, h);

    thrust::constant_iterator<unsigned> it_spid(spid);
    append(this->d_spid, numParticlesToAdd, spid);

    // assign a unique ID to each particle
    this->d_id.resize(numParticlesBefore + numParticlesToAdd);
    thrust::sequence(thrust::device, this->d_id.data() + numParticlesBefore, this->d_id.data() + this->d_id.size(),
                     numParticlesBefore);
}

void ParticleList::add(cstone::DeviceVector<scalar>& d_x, cstone::DeviceVector<scalar>& d_y,
                       cstone::DeviceVector<scalar>& d_z, scalar h, scalar q, scalar m, unsigned spid)
{
    int np = d_x.size();
	cstone::DeviceVector<scalar> d_vx(np);
	cstone::DeviceVector<scalar> d_vy(np);
	cstone::DeviceVector<scalar> d_vz(np);
	add(d_x, d_y, d_z, d_vx, d_vy, d_vz, h, q, m, spid);
}

void ParticleList::correctField()
{
    thrust::constant_iterator<scalar> it_coulomb(-Constants::COULOMB);
    thrust::transform(thrust::device, d_ex.data(), d_ex.data() + d_ex.size(), it_coulomb, d_ex.data(),
                      thrust::multiplies<scalar>());
    thrust::transform(thrust::device, d_ey.data(), d_ey.data() + d_ey.size(), it_coulomb, d_ey.data(),
                      thrust::multiplies<scalar>());
    thrust::transform(thrust::device, d_ez.data(), d_ez.data() + d_ez.size(), it_coulomb, d_ez.data(),
                      thrust::multiplies<scalar>());
}

/*void ParticleList::test()
{
    thrust::host_vector<scalar> h_x = d_x;
    thrust::host_vector<scalar> h_y = d_y;
    thrust::host_vector<scalar> h_z = d_z;

    printf("%f %f %f\n", h_x[100], h_y[100], h_z[100]);
}*/

std::array<scalar, 6> ParticleList::getBounds()
{
    std::array<scalar, 6> ret{0, 0, 0, 0, 0, 0};
    if (size() == 0) { return ret; }

    std::tie(ret[0], ret[1]) = cstone::MinMaxGpu<scalar>()(d_x.data(), d_x.data() + d_x.size());
    std::tie(ret[2], ret[3]) = cstone::MinMaxGpu<scalar>()(d_y.data(), d_y.data() + d_y.size());
    std::tie(ret[4], ret[5]) = cstone::MinMaxGpu<scalar>()(d_z.data(), d_z.data() + d_z.size());

    return ret;
}

} // namespace chora
