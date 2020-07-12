#pragma once

#include <iostream>
#include "core.hpp"

namespace dz
{

// NdArray�̏o�̓w���p�[�N���X
class NdArrayPrinter
{
public:
	NdArray& data;
	NdArrayPrinter(NdArray& data) :
		data(data)
	{}
};

extern std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda);

}
