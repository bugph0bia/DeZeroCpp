#pragma once

#include <iostream>
#include "core.hpp"

namespace dz
{

// NdArray�̏o�̓w���p�[�N���X
class NdArrayPrinter
{
public:
	const std::shared_ptr<NdArray> data;
	NdArrayPrinter(const std::shared_ptr<NdArray>& data) :
		data(data)
	{}
	NdArrayPrinter(const NdArray& data) :
		data(std::make_shared<NdArray>(data))
	{}
};

extern std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda);

}
