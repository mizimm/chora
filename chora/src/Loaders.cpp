#include "Loaders.h"

#include <array>
#include <cmath>

#include <thrust/host_vector.h>

#include "Random.h"
#include "ErfInv.h"

namespace chora
{

void Loaders::loadMaxwellianSphere(Maxwellian* f, scalar h, scalar q, scalar m, scalar np2c, int spid, chora::ParticleList* plist, unsigned npsphere, scalar radius)
{
	unsigned npbox = 6. / M_PI * (scalar)npsphere;

	std::array<scalar, 6> bounds{-radius, radius, -radius, radius, -radius, radius};

	thrust::host_vector<scalar> h_x;
	thrust::host_vector<scalar> h_y;
	thrust::host_vector<scalar> h_z;
	thrust::host_vector<scalar> h_vx;
	thrust::host_vector<scalar> h_vy;
	thrust::host_vector<scalar> h_vz;

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

	thrust::device_vector<scalar> d_x = h_x;
	thrust::device_vector<scalar> d_y = h_y;
	thrust::device_vector<scalar> d_z = h_z;
	thrust::device_vector<scalar> d_vx = h_vx;
	thrust::device_vector<scalar> d_vy = h_vy;
	thrust::device_vector<scalar> d_vz = h_vz;
	plist->add(d_x, d_y, d_z, d_vx, d_vy, d_vz, h, q*np2c, m*np2c, spid);
}

}
