
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step03 {

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
	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	Variable operator()(const Variable& input)
	{
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

//----------------------------------
// function
//----------------------------------

void step03()
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);
	std::cout << NdArrayPrinter(y.data) << std::endl;
}

}
