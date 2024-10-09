// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include "Random.h"

#include <cstdlib>

namespace chora
{

double Random::one()
{
	return (double)rand() / RAND_MAX;
}

double Random::pmone()
{
	return one() * 2 - 1;
}

double Random::pmhalf()
{
	return one() - 0.5;
}

double Random::between(double a, double b)
{
	return a + (b-a) * one();
}

}
