// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#include "Random.h"

#include <cstdlib>

namespace chora
{

scalar Random::one()
{
	return (double)rand() / RAND_MAX;
}

scalar Random::pmone()
{
	return one() * 2 - 1;
}

scalar Random::pmhalf()
{
	return one() - 0.5;
}

scalar Random::between(scalar a, scalar b)
{
	return a + (b-a) * one();
}

}
