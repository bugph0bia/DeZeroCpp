#pragma once

#include <iostream>
#include <cmath>
#include "NumCpp.hpp"

namespace dz
{

//----------------------------------
// typedef
//----------------------------------
using data_t = double;	// TODO: 最終的には float にする
using NdArray = nc::NdArray<data_t>;

//----------------------------------
// utility
//----------------------------------
// NdArrayの出力ヘルパークラス
class NdArrayPrinter
{
public:
	NdArray& data;
	NdArrayPrinter(NdArray& data) :
		data(data)
	{}
};
std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// NdArrayがスカラーなら中身のデータを標準出力へ
	if (nda.data.shape().rows == 1 && nda.data.shape().cols == 1) ost << nda.data[0];
	else ost << nda.data;
	return ost;
}

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
	{}

	// デストラクタ
	virtual ~Variable() {}
};

// 関数クラス
class Function
{
public:
	// ()演算子
	Variable operator()(const Variable& input) {
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		return output;
	}

	// 順伝播
	virtual NdArray forward(const NdArray& x) = 0;
};

// 関数クラス（2乗）
class Square : public Function
{
public:
	// 順伝播
	NdArray forward(const NdArray& x) override
	{
		return nc::power(x, 2);
	}
};

// 関数クラス（exp）
class Exp : public Function
{
public:
	// 順伝播
	NdArray forward(const NdArray& x) override
	{
		return nc::exp(x);
	}
};

}	// namespace dezerocpp
