
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step06 {

//----------------------------------
// class
//----------------------------------

// 変数クラス
class Variable
{
public:
	// 内部データ
	NdArray	data;
	// 勾配
	std::shared_ptr<NdArray> grad;

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
protected:
	// 入力データ
	std::shared_ptr<Variable> input;

public:
	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	Variable operator()(const Variable& input)
	{
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		this->input = std::make_shared<Variable>(input);
		return output;
	}

	// 順伝播
	virtual NdArray forward(const NdArray& x) = 0;
	// 逆伝播
	virtual NdArray backward(const NdArray& gy) = 0;
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
	// 逆伝播
	NdArray backward(const NdArray& gy) override
	{
		auto x = this->input->data;
		auto gx = 2.0 * x * gy;
		return gx;
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
	// 逆伝播
	NdArray backward(const NdArray& gy) override
	{
		auto x = this->input->data;
		auto gx = nc::exp(x) * gy;
		return gx;
	}
};

//----------------------------------
// function
//----------------------------------

void step06()
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);

	y.grad = std::make_shared<NdArray>(NdArray({ 1.0 }));
	b.grad = std::make_shared<NdArray>(C.backward(*y.grad));
	a.grad = std::make_shared<NdArray>(B.backward(*b.grad));
	x.grad = std::make_shared<NdArray>(A.backward(*a.grad));

	std::cout << NdArrayPrinter(*x.grad);
}

}
