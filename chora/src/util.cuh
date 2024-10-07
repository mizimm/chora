// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include "cstone/cuda/cuda_utils.cuh"
#include "ParticleList.cuh"

namespace chora
{

/*template <typename T>
void applyFilter(thrust::device_vector<T>& d_in, thrust::device_vector<bool>& d_filter, thrust::device_vector<T>&
d_out);

template <typename T>
void applyFilter(thrust::device_vector<T>& d_in, thrust::device_vector<bool>& d_filter);
*/

// write values in d_a where d_filter is true
template<typename T>
extern void writeDeviceVectorBin(FILE* fid, cstone::DeviceVector<T>& d_a, cstone::DeviceVector<bool>& d_filter);

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

} // namespace chora
