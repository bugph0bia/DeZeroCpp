
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step11 {

class Variable;
class Function;

//----------------------------------
// typedef
//----------------------------------
// スマートポインタ型
using NdArrayPtr = std::shared_ptr<NdArray>;	// インスタンス生成時は std::make_shared<NdArray> 関数を使うこと
using VariablePtr = std::shared_ptr<Variable>;	// インスタンス生成時は std::make_shared<Variable> 関数を使うこと
using FunctionPtr = std::shared_ptr<Function>;	// 派生クラスのインスタンス生成時は new を使うこと
												// （make_shared を使うと Function クラスがインスタンス化されてエラーとなる）
// リスト型
using NdArrayPtrList = std::vector<NdArrayPtr>;
using VariablePtrList = std::vector<VariablePtr>;


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
	VariablePtrList inputs;
	// 出力データ
	VariablePtrList outputs;

	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	VariablePtrList operator()(const VariablePtr& input)
	{
		return (*this)(VariablePtrList({ input }));
	}

	// ()演算子
	VariablePtrList operator()(const VariablePtrList& inputs)
	{
		auto xs = NdArrayPtrList();
		for(const auto& i : inputs) {
			xs.push_back(i->data);
		}

		auto ys = this->forward(xs);
		auto outputs = VariablePtrList();
		for(const auto& y : ys) {
			auto o = as_variable(as_array(*y));
			o->set_creator(shared_from_this());
			outputs.push_back(o);
		}

		this->inputs = std::move(inputs);
		this->outputs = std::move(outputs);
		return this->outputs;
	}

	// 順伝播
	virtual NdArrayPtrList forward(const NdArrayPtrList& xs) = 0;
	// 逆伝播
	virtual NdArrayPtr backward(const NdArrayPtr& gy) = 0;
};

// 逆伝播
// 内部で Function クラスのメンバを参照しているためこの位置で定義する必要がある
void Variable::backward()
{
	//if (!this->grad) {
	//	// 勾配の初期値(1)を設定
	//	auto g = nc::ones_like<data_t>(*this->data);
	//	this->grad = as_array(g);
	//}

	//// 関数リスト
	//auto funcs = std::list<FunctionPtr>({ this->creator });
	//while (!funcs.empty()) {
	//	// リストから関数を取り出す
	//	auto f = funcs.back();
	//	funcs.pop_back();
	//	// 関数の入出力を取得
	//	auto x = f->input;
	//	auto y = f->output;
	//	// 逆伝播を呼ぶ
	//	x->grad = f->backward(y->grad);

	//	if (x->creator) {
	//		// １つ前の関数をリストに追加
	//		funcs.push_back(x->creator);
	//	}
	//}
}

// 関数クラス（加算）
class Add : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = xs[0];
		auto x1 = xs[1];
		auto y = (*x0) + (*x1);
		return NdArrayPtrList({ as_array(y) });
	}
	// 逆伝播
	NdArrayPtr backward(const NdArrayPtr& gy) override
	{
		// 暫定
		return gy;
	}
};

//----------------------------------
// function
//----------------------------------

void step11()
{
	auto xs = VariablePtrList({ as_variable(as_array({ 2.0 })), as_variable(as_array({ 3.0 })) });
	auto f = std::shared_ptr<Function>(new Add());
	auto ys = (*f)(xs);
	auto y = ys[0];
	std::cout << NdArrayPrinter(*y->data) << std::endl;
}

}
