// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

namespace chora
{

class Random
{
public:

	static double one();
	static double pmone();
	static double pmhalf();
	static double between(double a, double b);
};

}
