// © 2024 Mike Zimmerman, The Johns Hopkins University Applied Physics Laboratory LLC

#pragma once

namespace chora
{

class ErfInv
{
public:
	static const double ERFINVCONST;
	static double value(const double& x);
};

}
