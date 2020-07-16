#pragma once

#include "pch.h"

#include <iostream>

namespace dz
{

std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// nullptr の場合
	if (!nda.data) ost << "Null";
	// NdArrayがスカラーなら中身のデータを標準出力へ
	else if (nda.data->shape().rows == 1 && nda.data->shape().cols == 1) ost << (*nda.data)[0];
	// 通常時
	else ost << nda.data;
	return ost;
}

}
