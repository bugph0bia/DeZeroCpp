
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step08 {

//----------------------------------
// class
//----------------------------------
class Variable;
class Function;

using NdArrayPtr = std::shared_ptr<NdArray>;
using VariablePtr = std::shared_ptr<Variable>;
using FunctionPtr = std::shared_ptr<Function>;


// 変数クラス
class Variable
{
	//protected:	// このステップでは一時的にpublicにする
public:
	// 勾配
	NdArrayPtr grad;
	// 生成元の関数
	FunctionPtr creator;

public:
	// 内部データ
	NdArray	data;

	// コンストラクタ
	Variable(const NdArray& data) :
		data(data)
	{}

	// デストラクタ
	virtual ~Variable() {}

	// 生成元の関数を設定
	void set_creator(const FunctionPtr& func)
	{
		creator = func;
	}

	// 逆伝播(再帰)
	void backward();
};

// 関数クラス
class Function : public std::enable_shared_from_this<Function>
{
	//protected:	// このステップでは一時的にpublicにする
public:
	// 入力データ
	VariablePtr input;
	// 出力データ
	VariablePtr output;

public:
	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	VariablePtr operator()(const VariablePtr& input)
	{
		auto x = input->data;
		auto y = this->forward(x);
		auto output = std::make_shared<Variable>(y);
		output->set_creator(shared_from_this());
		this->input = input;
		this->output = output;
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

// 逆伝播(再帰)
// 内部で Function クラスのメンバを参照しているためこの位置で定義する必要がある
void Variable::backward()
{
	// 関数リスト
	auto funcs = std::vector<FunctionPtr>({ this->creator });
	while (!funcs.empty()) {
		// リストから関数を取り出す
		auto f = funcs.back();
		funcs.pop_back();
		// 関数の入出力を取得
		auto x = f->input;
		auto y = f->output;
		// 逆伝播を呼ぶ
		x->grad = std::make_shared<NdArray>(f->backward(*y->grad));

		if (x->creator != nullptr) {
			// １つ前の関数をリストに追加
			funcs.push_back(x->creator);
		}
	}
}

//----------------------------------
// function
//----------------------------------

void step08()
{
	auto A = FunctionPtr(new Square());
	auto B = FunctionPtr(new Exp());
	auto C = FunctionPtr(new Square());

	auto x = std::make_shared<Variable>(NdArray({ 0.5 }));
	auto a = (*A)(x);
	auto b = (*B)(a);
	auto y = (*C)(b);

	y->grad = std::make_shared<NdArray>(NdArray({ 1.0 }));
	y->backward();
	std::cout << NdArrayPrinter(*x->grad) << std::endl;
}

}
