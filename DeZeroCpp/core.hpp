#pragma once

#include "NumCpp.hpp"

namespace dz
{

//----------------------------------
// typedef
//----------------------------------
using data_t = float;
using NdArray = nc::NdArray<data_t>;

//----------------------------------
// class
//----------------------------------
// 変数クラス
class Variable
{
public:
	// 内部データ
	NdArray	data;

	// コンストラクタ
	Variable(const NdArray& data) :
		data(data)
	{
	}

	// デストラクタ
	virtual ~Variable() {}
};

}	// namespace dezerocpp
