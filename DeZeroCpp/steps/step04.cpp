
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step04 {

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

// 数値微分
NdArray numerical_diff(std::function<Variable(Variable)> f, const Variable& x, data_t eps = 1e-4)
{
	auto x0 = Variable(x.data - eps);
	auto x1 = Variable(x.data + eps);
	auto y0 = f(x0);
	auto y1 = f(x1);
	return (y1.data - y0.data) / (2 * eps);
}

Variable f(const Variable& x)
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();
	return C(B(A(x)));
}

void step04()
{
	auto x = Variable(NdArray({ 0.5 }));
	auto dy = numerical_diff(f, x);
	std::cout << NdArrayPrinter(dy) << std::endl;
}

}
