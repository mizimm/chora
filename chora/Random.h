// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

#include "scalar.h"

namespace chora
{

class Random
{
public:

	static scalar one();
	static scalar pmone();
	static scalar pmhalf();
	static scalar between(scalar a, scalar b);
};

}
