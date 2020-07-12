#pragma once

#include <iostream>
#include "core.hpp"

namespace dz
{

// NdArrayの出力ヘルパークラス
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
