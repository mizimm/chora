// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
//#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include "util.cuh"

namespace chora
{

/*template <typename T>
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
*/
void writeParticlesBin(ParticleList* plist, std::string filename)
{

	// filter on species, cf. https://stackoverflow.com/questions/20071454/thrust-gathering-filtering
	thrust::device_vector<bool> d_filter(plist->size());	// this will contain 1 where the species matches
//	thrust::device_vector<Species*> d_sp = plist->h_sp;
//	thrust::transform(d_sp.begin(), d_sp.end(), d_filter.begin(), IsSpecies(sp));
	thrust::fill(d_filter.begin(), d_filter.end(), true);

	// write file
	FILE* fid = fopen(filename.c_str(), "wb");
	int np = thrust::count(d_filter.begin(), d_filter.end(), 1);
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

/*NormalSampler::NormalSampler(double a, double b)
	:
	a(a),
	b(b)
{
}

double NormalSampler::operator()(const unsigned n) const
{
	thrust::default_random_engine rng;
	thrust::random::normal_distribution<double> dist(a, b);
	rng.discard(n);
	return dist(rng);
}

UniformSampler::UniformSampler(double a, double b)
	:
	a(a),
	b(b)
{
}

double UniformSampler::operator()(const unsigned n) const
{
	thrust::default_random_engine rng;
	thrust::random::uniform_real_distribution<double> dist(a, b);
	rng.discard(n);
	return dist(rng);
}*/

}
