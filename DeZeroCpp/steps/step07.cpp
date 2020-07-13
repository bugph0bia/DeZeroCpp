
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step07 {

//----------------------------------
// class
//----------------------------------
class Variable;
class Function;

// 変数クラス
class Variable
{
//protected:	// このステップでは一時的にpublicにする
public:
	// 勾配
	NdArray* grad;
	// 生成元の関数
	Function* creator;

public:
	// 内部データ
	NdArray	data;

	// コンストラクタ
	Variable(const NdArray& data) :
		data(data)
	{}

	// デストラクタ
	virtual ~Variable()
	{
		if (grad) delete grad;
	}

	// 生成元の関数を設定
	void set_creator(Function* func)
	{
		creator = func;
	}

	// 逆伝播(再帰)
	void backward();
};

// 関数クラス
class Function
{
//protected:	// このステップでは一時的にpublicにする
public:
	// 入力データ
	Variable* input;
	// 出力データ
	Variable* output;

public:
	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	Variable operator()(Variable& input)
	{
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		output.set_creator(this);
		this->input = &input;
		this->output = &output;
		return output;
	}

	// 順伝播
	virtual NdArray forward(const NdArray& x)
	{
		// No Implemented
		assert(false);
		return x;
	}
	// 逆伝播
	virtual NdArray backward(const NdArray& gy)
	{
		// No Implemented
		assert(false);
		return gy;
	}
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

// 逆伝播(再帰)
// 内部で Function クラスのメンバを参照しているためこの位置で定義する必要がある
void Variable::backward()
{
	auto f = this->creator;
	if (f != nullptr) {
		auto x = f->input;
		x->grad = new NdArray(f->backward(*(this->grad)));
		x->backward();
	}
}

//----------------------------------
// function
//----------------------------------

void step07()
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);

	//assert(y.creator == &C);
	//assert(y.creator->input == &b);
	//assert(y.creator->input->creator == &B);
	//assert(y.creator->input->creator->input == &a);
	//assert(y.creator->input->creator->input->creator == &A);
	//assert(y.creator->input->creator->input->creator->input == &x);

	//y.grad = new NdArray({ 1.0 });

	//auto tmp_C = y.creator;
	//auto tmp_b = tmp_C->input;
	//tmp_b->grad = new NdArray(tmp_C->backward(*y.grad));

	//auto tmp_B = tmp_b->creator;
	//auto tmp_a = tmp_B->input;
	//tmp_a->grad = new NdArray(tmp_B->backward(*tmp_b->grad));

	//auto tmp_A = tmp_a->creator;
	//auto tmp_x = tmp_A->input;
	//tmp_x->grad = new NdArray(tmp_A->backward(*tmp_a->grad));

	//std::cout << NdArrayPrinter(*tmp_x->grad) << std::endl;

	y.grad = new NdArray({ 1.0 });
	y.backward();
	std::cout << NdArrayPrinter(*x.grad) << std::endl;
}

}
