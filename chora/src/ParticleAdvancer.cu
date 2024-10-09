// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include "ParticleAdvancer.cuh"

namespace chora
{

ParticleAdvancer::ParticleAdvancer(double dt) :
	dt(dt)
{
}

__host__ __device__ thrust::tuple<double,double,double,double,double,double> ParticleAdvancer::operator ()(thrust::tuple<double,double,double,double,double,double,double,double,double,double,double> t)
{
	double x = thrust::get<0>(t);
	double y = thrust::get<1>(t);
	double z = thrust::get<2>(t);
	double vx = thrust::get<3>(t);
	double vy = thrust::get<4>(t);
	double vz = thrust::get<5>(t);
	double ex = thrust::get<6>(t);
	double ey = thrust::get<7>(t);
	double ez = thrust::get<8>(t);
	double q = thrust::get<9>(t);
	double m = thrust::get<10>(t);

//	vx += ex * q / m * dt;
//	vy += ey * q / m * dt;
//	vz += ex * q / m * dt;

	vx += ex * q / m * dt;
	vy += ey * q / m * dt;
	vz += ez * q / m * dt;

	x += vx * dt;
	y += vy * dt;
	z += vz * dt;
	return thrust::make_tuple(x, y, z, vx, vy, vz);
}

void ParticleAdvancer::apply(ParticleList* plist, double dt)
{
	thrust::device_vector<double> xold = plist->d_x;
	thrust::device_vector<double> yold = plist->d_y;
	thrust::device_vector<double> zold = plist->d_z;

	auto beginOld = thrust::make_zip_iterator(
		thrust::make_tuple(
			plist->d_x.begin(),
			plist->d_y.begin(),
			plist->d_z.begin(),
			plist->d_vx.begin(),
			plist->d_vy.begin(),
			plist->d_vz.begin(),
			plist->d_ex.begin(),
			plist->d_ey.begin(),
			plist->d_ez.begin(),
			plist->d_q.begin(),
			plist->d_m.begin()
		)
	);

	auto endOld = thrust::make_zip_iterator(
		thrust::make_tuple(
			plist->d_x.end(),
			plist->d_y.end(),
			plist->d_z.end(),
			plist->d_vx.end(),
			plist->d_vy.end(),
			plist->d_vz.end(),
			plist->d_ex.end(),
			plist->d_ey.end(),
			plist->d_ez.end(),
			plist->d_q.end(),
			plist->d_m.end()
		)
	);

	auto beginNew = thrust::make_zip_iterator(
		thrust::make_tuple(
			plist->d_x.begin(),
			plist->d_y.begin(),
			plist->d_z.begin(),
			plist->d_vx.begin(),
			plist->d_vy.begin(),
			plist->d_vz.begin()
		)
	);

        thrust::transform(
                beginOld, //thrust::make_zip_iterator( thrust::make_tuple( plist->d_x.begin(), plist->d_y.begin(), plist->d_z.begin(), plist->d_vx.begin(), plist->d_vy.begin(), plist->d_vz.begin(), plist->d_ex.begin(), plist->d_ey.begin(), plist->d_ex.begin(), plist->d_q.begin(), plist->d_m.begin() ) ),
                endOld, //thrust::make_zip_iterator( thrust::make_tuple( plist->d_x.end(),   plist->d_y.end(),   plist->d_z.end(),   plist->d_vx.end(),   plist->d_vy.end(),   plist->d_vz.end(), plist->d_ex.end(), plist->d_ey.end(), plist->d_ex.end(), plist->d_q.end(), plist->d_m.end() ) ),
                beginNew, //thrust::make_zip_iterator( thrust::make_tuple( plist->d_x.begin(), plist->d_y.begin(), plist->d_z.begin(), plist->d_vx.begin(), plist->d_vy.begin(),  plist->d_vz.begin() ) ),
                *this
        );

	// TODO: add particle-surface hits using xold, yold, zold
}

}
