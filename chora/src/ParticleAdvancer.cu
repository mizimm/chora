// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include "ParticleAdvancer.cuh"

#include "thrust/iterator/zip_iterator.h"
#include "thrust/transform.h"
#include "thrust/execution_policy.h"

namespace chora
{

ParticleAdvancer::ParticleAdvancer(scalar dt) :
	dt(dt)
{
}

__host__ __device__ thrust::tuple<scalar,scalar,scalar,scalar,scalar,scalar> ParticleAdvancer::operator ()(thrust::tuple<scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar> t)
{
	scalar x = thrust::get<0>(t);
	scalar y = thrust::get<1>(t);
	scalar z = thrust::get<2>(t);
	scalar vx = thrust::get<3>(t);
	scalar vy = thrust::get<4>(t);
	scalar vz = thrust::get<5>(t);
	scalar ex = thrust::get<6>(t);
	scalar ey = thrust::get<7>(t);
	scalar ez = thrust::get<8>(t);
	scalar q = thrust::get<9>(t);
	scalar m = thrust::get<10>(t);

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
	auto xold = plist->d_x;
	auto yold = plist->d_y;
	auto zold = plist->d_z;

	auto beginOld = thrust::make_zip_iterator(
		thrust::make_tuple(
			plist->d_x.data(),
			plist->d_y.data(),
			plist->d_z.data(),
			plist->d_vx.data(),
			plist->d_vy.data(),
			plist->d_vz.data(),
			plist->d_ex.data(),
			plist->d_ey.data(),
			plist->d_ez.data(),
			plist->d_q.data(),
			plist->d_m.data()
		)
	);

    auto numElements = plist->d_x.size();
	auto endOld = thrust::make_zip_iterator(
		thrust::make_tuple(
			plist->d_x.data() + numElements,
			plist->d_y.data() + numElements,
			plist->d_z.data() + numElements,
			plist->d_vx.data() + numElements,
			plist->d_vy.data() + numElements,
			plist->d_vz.data() + numElements,
			plist->d_ex.data() + numElements,
			plist->d_ey.data() + numElements,
			plist->d_ez.data() + numElements,
			plist->d_q.data() + numElements,
			plist->d_m.data() + numElements
		)
	);

	auto beginNew = thrust::make_zip_iterator(
		thrust::make_tuple(
			plist->d_x.data(),
			plist->d_y.data(),
			plist->d_z.data(),
			plist->d_vx.data(),
			plist->d_vy.data(),
			plist->d_vz.data()
		)
	);

        thrust::transform(thrust::device,
                beginOld, //thrust::make_zip_iterator( thrust::make_tuple( plist->d_x.begin(), plist->d_y.begin(), plist->d_z.begin(), plist->d_vx.begin(), plist->d_vy.begin(), plist->d_vz.begin(), plist->d_ex.begin(), plist->d_ey.begin(), plist->d_ex.begin(), plist->d_q.begin(), plist->d_m.begin() ) ),
                endOld, //thrust::make_zip_iterator( thrust::make_tuple( plist->d_x.end(),   plist->d_y.end(),   plist->d_z.end(),   plist->d_vx.end(),   plist->d_vy.end(),   plist->d_vz.end(), plist->d_ex.end(), plist->d_ey.end(), plist->d_ex.end(), plist->d_q.end(), plist->d_m.end() ) ),
                beginNew, //thrust::make_zip_iterator( thrust::make_tuple( plist->d_x.begin(), plist->d_y.begin(), plist->d_z.begin(), plist->d_vx.begin(), plist->d_vy.begin(),  plist->d_vz.begin() ) ),
                *this
        );

	// TODO: add particle-surface hits using xold, yold, zold
}

}
