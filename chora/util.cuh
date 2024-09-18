// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>

#include "ParticleList.cuh"

namespace chora
{

/*template <typename T>
void applyFilter(thrust::device_vector<T>& d_in, thrust::device_vector<bool>& d_filter, thrust::device_vector<T>& d_out);

template <typename T>
void applyFilter(thrust::device_vector<T>& d_in, thrust::device_vector<bool>& d_filter);

template <typename T>
void writeDeviceVectorBin(FILE* fid, thrust::device_vector<T>& d_a, thrust::device_vector<bool>& d_filter); // write values in d_a where d_filter is true
*/

template <typename T>
void applyFilter(thrust::device_vector<T>& d_in, thrust::device_vector<bool>& d_filter, thrust::device_vector<T>& d_out)        // apply precomputed filter flags to d_in, to produce d_out containing all value$
{
        thrust::device_vector<T> d_temp(d_in.size());
        auto end = thrust::copy_if(d_in.begin(), d_in.end(), d_filter.begin(), d_temp.begin(), thrust::identity<bool>() );    // for a bool, the identity is its own truth value (iden$
        int len = end - d_temp.begin();
        d_out.resize(len);
        thrust::copy_n(d_temp.begin(), len, d_out.begin());
}

template <typename T>
void applyFilter(thrust::device_vector<T>& d_in, thrust::device_vector<bool>& d_filter)
{
	applyFilter(d_in, d_filter, d_in);
}


template <typename T>
void writeDeviceVectorBin(FILE* fid, thrust::device_vector<T>& d_a, thrust::device_vector<bool>& d_filter)	// write values in d_a where d_filter is true
{
	thrust::device_vector<T> d_tmp;
	applyFilter(d_a, d_filter, d_tmp);
	thrust::host_vector<T> h_tmp = d_tmp;
	fwrite(h_tmp.data(), sizeof(T), h_tmp.size(), fid);
}

void writeParticlesBin(ParticleList* plist, std::string filename);


struct NormalSampler
{
	scalar a, b;

	__host__ __device__ NormalSampler(scalar a, scalar b);

	__host__ __device__ scalar operator()(const unsigned n) const;
};

struct UniformSampler
{
	scalar a, b;

	__host__ __device__ UniformSampler(scalar a, scalar b);

	__host__ __device__ scalar operator()(const unsigned n) const;
};


}
