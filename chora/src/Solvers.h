// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC
// © 2024 Sebastian Keller

#include "ParticleList.cuh"

namespace chora
{

class Solvers
{
public:
	static void multipoleGpu(chora::ParticleList* plist);
	static void directGpu(chora::ParticleList* plist);
	static void directCpu(chora::ParticleList* plist, scalar dmin);
};

}
