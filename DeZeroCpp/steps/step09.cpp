
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step09 {

class Variable;
class Function;

//----------------------------------
// typedef
//----------------------------------
// NdArrayクラスのスマートポインタ型
// インスタンス生成時は std::make_shared<NdArray> 関数を使うこと
using NdArrayPtr = std::shared_ptr<NdArray>;

// Variableクラスのスマートポインタ型
// インスタンス生成時は std::make_shared<Variable> 関数を使うこと
using VariablePtr = std::shared_ptr<Variable>;

// Functionクラスのスマートポインタ型
// 派生クラスのインスタンス生成時は new を使うこと
// （make_shared を使うと Function クラスがインスタンス化されてエラーとなる）
using FunctionPtr = std::shared_ptr<Function>;

//// std::initializer_list の {...} 形式で std::make_shared するためのヘルパー関数
//template<typename ObjType, typename DataType>
//std::shared_ptr<ObjType> make_shared_from_list(std::initializer_list<DataType> list) {
//	return std::make_shared<ObjType>(std::move(list));
//}

// NdArrayPtr作成
NdArrayPtr as_array(nullptr_t = nullptr)
{
	return NdArrayPtr();	// 引数なしまたは nullptr の場合は Empty とする
}
NdArrayPtr as_array(std::initializer_list<NdArray::value_type> list)
{
	return std::make_shared<NdArray>(list);
}
NdArrayPtr as_array(NdArray::value_type scalar)
{
	return as_array({ scalar });
}
NdArrayPtr as_array(const NdArray& data)
{
	return std::make_shared<NdArray>(data);
}

// VariablePtr作成
VariablePtr as_variable(const NdArrayPtr& data)
{
	return std::make_shared<Variable>(data);
}
VariablePtr as_variable(const Variable& data)
{
	return std::make_shared<Variable>(data);
}

//----------------------------------
// class
//----------------------------------

// 変数クラス
class Variable
{
public:
	// 内部データ
	NdArrayPtr data;
	// 勾配
	NdArrayPtr grad;
	// 生成元の関数
	FunctionPtr creator;

	// コンストラクタ
	Variable(const NdArrayPtr& data) :
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
public:
	// 入力データ
	VariablePtr input;
	// 出力データ
	VariablePtr output;

	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	VariablePtr operator()(const VariablePtr& input)
	{
		auto x = input->data;
		auto y = this->forward(x);
		auto output = as_variable(y);
		output->set_creator(shared_from_this());
		this->input = input;
		this->output = output;
		return output;
	}

	// 順伝播
	virtual NdArrayPtr forward(const NdArrayPtr& px) = 0;
	// 逆伝播
	virtual NdArrayPtr backward(const NdArrayPtr& pgy) = 0;
};

// 関数クラス（2乗）
class Square : public Function
{
public:
	// 順伝播
	NdArrayPtr forward(const NdArrayPtr& px) override
	{
		auto x = *px;
		auto y = nc::power(x, 2);
		return as_array(y);
	}
	// 逆伝播
	NdArrayPtr backward(const NdArrayPtr& pgy) override
	{
		auto x = *this->input->data;
		auto gy = *pgy;
		auto gx = 2.0 * x * gy;
		return as_array(gx);
	}
};

// 関数クラス（exp）
class Exp : public Function
{
public:
	// 順伝播
	NdArrayPtr forward(const NdArrayPtr& px) override
	{
		auto x = *px;
		auto y = nc::exp(x);
		return as_array(y);
	}
	// 逆伝播
	NdArrayPtr backward(const NdArrayPtr& pgy) override
	{
		auto x = *this->input->data;
		auto gy = *pgy;
		auto gx = nc::exp(x) * gy;
		return as_array(gx);
	}
};

// 逆伝播
// 内部で Function クラスのメンバを参照しているためこの位置で定義する必要がある
void Variable::backward()
{
	if (!this->grad) {
		// 勾配の初期値(1)を設定
		auto g = nc::ones_like<data_t>(*this->data);
		this->grad = as_array(g);
	}

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
		x->grad = f->backward(y->grad);

		if (x->creator) {
			// １つ前の関数をリストに追加
			funcs.push_back(x->creator);
		}
	}
}

//----------------------------------
// function
//----------------------------------
VariablePtr square(VariablePtr x)
{
	auto f = FunctionPtr(new Square());
	return (*f)(x);
}

VariablePtr exp(VariablePtr x)
{
	auto f = FunctionPtr(new Exp());
	return (*f)(x);
}

void step09()
{
	auto x = as_variable(as_array({ 0.5 }));
	//auto a = square(x);
	//auto b = exp(a);
	//auto y = square(b);
	auto y = square(exp(square(x)));

	y->backward();
	std::cout << NdArrayPrinter(x->grad) << std::endl;
}

}
