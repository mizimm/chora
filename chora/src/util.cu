// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include "scalar.h"
#include "util.cuh"

namespace chora
{

template<typename T>
void applyFilter(
    cstone::DeviceVector<T>& d_in, cstone::DeviceVector<bool>& d_filter,
    cstone::DeviceVector<T>& d_out) // apply precomputed filter flags to d_in, to produce d_out containing all value$
{
    cstone::DeviceVector<T> d_temp(d_in.size());
    auto end = thrust::copy_if(thrust::device, d_in.data(), d_in.data() + d_in.size(), d_filter.data(), rawPtr(d_temp),
                                                  thrust::identity<bool>()); // for a bool, the identity is its own truth value (iden$
    int  len = end - d_temp.data();
    d_out.resize(len);
    thrust::copy_n(thrust::device, d_temp.data(), len, d_out.data());
}

template<typename T>
void applyFilter(cstone::DeviceVector<T>& d_in, cstone::DeviceVector<bool>& d_filter)
{
    applyFilter(d_in, d_filter, d_in);
}

template<typename T>
void writeDeviceVectorBin(FILE* fid, cstone::DeviceVector<T>& d_a,
                          cstone::DeviceVector<bool>& d_filter) // write values in d_a where d_filter is true
{
    cstone::DeviceVector<T> d_tmp;
    applyFilter(d_a, d_filter, d_tmp);
    std::vector<T> h_tmp = toHost(d_tmp);
    fwrite(h_tmp.data(), sizeof(T), h_tmp.size(), fid);
}

#define WRITE_DEVICE_VECTOR_BIN(T) \
template void writeDeviceVectorBin(FILE*, cstone::DeviceVector<T>&, cstone::DeviceVector<bool>&)

WRITE_DEVICE_VECTOR_BIN(double);
WRITE_DEVICE_VECTOR_BIN(float);
WRITE_DEVICE_VECTOR_BIN(int);
WRITE_DEVICE_VECTOR_BIN(unsigned);

void writeParticlesBin(ParticleList* plist, std::string filename)
{
	// filter on species, cf. https://stackoverflow.com/questions/20071454/thrust-gathering-filtering
    cstone::DeviceVector<bool> d_filter(plist->size());	// this will contain 1 where the species matches
//	thrust::device_vector<Species*> d_sp = plist->h_sp;
//	thrust::transform(d_sp.begin(), d_sp.end(), d_filter.begin(), IsSpecies(sp));
	thrust::fill(thrust::device, d_filter.data(), d_filter.data() + d_filter.size(), true);

	// write file
	FILE* fid = fopen(filename.c_str(), "wb");
	int np = thrust::count(thrust::device, d_filter.data(), d_filter.data() + d_filter.size(), 1);
	fwrite(&np, sizeof(int), 1, fid);

//	std::cout << plist->d_q[0] << " " << plist->d_q[1] << " " << plist->d_ax[0] << " " << plist->d_ax[1] << std::endl;

	writeDeviceVectorBin(fid, plist->d_x, d_filter);
	writeDeviceVectorBin(fid, plist->d_y, d_filter);
	writeDeviceVectorBin(fid, plist->d_z, d_filter);
	writeDeviceVectorBin(fid, plist->d_vx, d_filter);
	writeDeviceVectorBin(fid, plist->d_vy, d_filter);
	writeDeviceVectorBin(fid, plist->d_vz, d_filter);
	writeDeviceVectorBin(fid, plist->d_ex, d_filter);
	writeDeviceVectorBin(fid, plist->d_ey, d_filter);
	writeDeviceVectorBin(fid, plist->d_ez, d_filter);
	writeDeviceVectorBin(fid, plist->d_spid, d_filter);
//	writeDeviceVectorBin(fid, plist->d_phi, d_filter);

/*	writeDeviceVectorBin(fid, plist->d_x);
	writeDeviceVectorBin(fid, plist->d_y);
	writeDeviceVectorBin(fid, plist->d_z);
	writeDeviceVectorBin(fid, plist->d_ax);
	writeDeviceVectorBin(fid, plist->d_ay);
	writeDeviceVectorBin(fid, plist->d_az);*/

	fclose(fid);
}

__host__ __device__ NormalSampler::NormalSampler(scalar a, scalar b)
	:
	a(a),
	b(b)
{
}

__host__ __device__ scalar NormalSampler::operator()(const unsigned n) const
{
	thrust::default_random_engine rng;
	thrust::random::normal_distribution<scalar> dist(a, b);
	rng.discard(n);
	return dist(rng);
}

__host__ __device__ UniformSampler::UniformSampler(scalar a, scalar b)
	:
	a(a),
	b(b)
{
}

__host__ __device__ scalar UniformSampler::operator()(const unsigned n) const
{
	thrust::default_random_engine rng;
	thrust::random::uniform_real_distribution<scalar> dist(a, b);
	rng.discard(n);
	return dist(rng);
}

}
