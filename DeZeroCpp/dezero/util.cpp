#pragma once

#include "pch.h"

#include <iostream>

namespace dz
{

std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// NdArrayがスカラーなら中身のデータを標準出力へ
	if (nda.data.shape().rows == 1 && nda.data.shape().cols == 1) ost << nda.data[0];
	else ost << nda.data;
	return ost;
}

}
