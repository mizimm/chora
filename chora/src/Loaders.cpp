#include "Loaders.h"

#include <array>
#include <cmath>

#include <thrust/host_vector.h>

#include "Random.h"
#include "ErfInv.h"

namespace chora
{

void Loaders::loadMaxwellianSphere(Maxwellian* f, double h, double q, double m, double np2c, int spid, chora::ParticleList* plist, unsigned npsphere, double radius)
{
	unsigned npbox = 6. / M_PI * (double)npsphere;

	std::array<double, 6> bounds{-radius, radius, -radius, radius, -radius, radius};

	thrust::host_vector<double> h_x;
	thrust::host_vector<double> h_y;
	thrust::host_vector<double> h_z;
	thrust::host_vector<double> h_vx;
	thrust::host_vector<double> h_vy;
	thrust::host_vector<double> h_vz;

	for (unsigned i = 0; i < npbox; i++)
	{
		double x = Random::between(-radius, radius);
		double y = Random::between(-radius, radius);
		double z = Random::between(-radius, radius);
		double r = sqrt(x*x + y*y + z*z);
		if (r > radius)
		{
			continue;
		}
		double vx = f->vdx + f->vth *  ErfInv::value(Random::pmone());
		double vy = f->vdy + f->vth *  ErfInv::value(Random::pmone());
		double vz = f->vdz + f->vth *  ErfInv::value(Random::pmone());
		//
		h_x.push_back(x);
		h_y.push_back(y);
		h_z.push_back(z);
		h_vx.push_back(vx);
		h_vy.push_back(vy);
		h_vz.push_back(vz);
	}

	thrust::device_vector<double> d_x = h_x;
	thrust::device_vector<double> d_y = h_y;
	thrust::device_vector<double> d_z = h_z;
	thrust::device_vector<double> d_vx = h_vx;
	thrust::device_vector<double> d_vy = h_vy;
	thrust::device_vector<double> d_vz = h_vz;
	plist->add(d_x, d_y, d_z, d_vx, d_vy, d_vz, h, q*np2c, m*np2c, spid);
}

}
