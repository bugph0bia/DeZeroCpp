
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step16 {

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
// nullptr代入で対応できるため削除
//NdArrayPtr as_array(nullptr_t = nullptr)
//{
//	return NdArrayPtr();	// 引数なしまたは nullptr の場合は Empty とする
//}
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
	// 世代
	int generation;

	// コンストラクタ
	Variable(const NdArrayPtr& data) :
		data(data),
		generation(0)
	{}

	// デストラクタ
	virtual ~Variable() {}

	// 生成元の関数を設定
	void set_creator(const FunctionPtr& func);

	// 逆伝播(再帰)
	void backward();

	// 微分を初期化
	void cleargrad() {
		this->grad = nullptr;
	}
};

// 関数クラス
class Function : public std::enable_shared_from_this<Function>
{
public:
	// 入力データ
	VariablePtrList inputs;
	// 出力データ
	VariablePtrList outputs;
	// 世代
	int generation;

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
		for (const auto& i : inputs) {
			xs.push_back(i->data);
		}

		auto ys = this->forward(xs);
		auto outputs = VariablePtrList();
		for (const auto& y : ys) {
			auto o = as_variable(as_array(*y));
			o->set_creator(shared_from_this());
			outputs.push_back(o);
		}
		
		auto max_elem = std::max_element(
			inputs.cbegin(), inputs.cend(),
			[](VariablePtr lhs, VariablePtr rhs) { return lhs->generation < rhs->generation; }
		);
		this->generation = (*max_elem)->generation;

		this->inputs = inputs;
		this->outputs = std::move(outputs);
		return this->outputs;
	}

	// 順伝播
	virtual NdArrayPtrList forward(const NdArrayPtrList& xs) = 0;
	// 逆伝播
	virtual NdArrayPtrList backward(const NdArrayPtrList& gy) = 0;
};

// 生成元の関数を設定
void Variable::set_creator(const FunctionPtr& func)
{
	creator = func;
	this->generation = func->generation + 1;
}

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
	auto funcs = std::list<FunctionPtr>();
	// 処理済み関数セット
	auto seen_set = std::set<FunctionPtr>();

	// クロージャ：関数リストへ追加
	auto add_func = [&funcs, &seen_set](const FunctionPtr& f) {
		// リストへ未追加の関数なら
		if (seen_set.find(f) == seen_set.end()) {
			// リストへ追加して世代で昇順ソートする
			funcs.push_back(f);
			seen_set.insert(f);
			funcs.sort([](const FunctionPtr& lhs, const FunctionPtr& rhs) { return lhs->generation < rhs->generation; });
		}
	};

	// 最初の関数をリストに追加
	add_func(this->creator);

	while (!funcs.empty()) {
		// リストから関数を取り出す
		auto f = funcs.back();
		funcs.pop_back();

		auto gys = NdArrayPtrList();
		for (const auto& o : f->outputs) {
			gys.push_back(o->grad);
		}
		auto gxs = f->backward(gys);

		assert(f->inputs.size() == gxs.size());

		for (size_t i = 0; i < gxs.size(); i++) {
			auto x = f->inputs[i];
			auto gx = gxs[i];

			if (!x->grad) {
				x->grad = gx;
			}
			else {
				// 新しい NdArrayPtr インスタンスを作ることが重要
				// 例えば、*x->grad += *gx; としてはいけない（付録A参照）
				x->grad = as_array(*x->grad + *gx);
			}

			if (x->creator) {
				// １つ前の関数をリストに追加
				add_func(x->creator);
			}
		}
	}
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
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		return { gys[0], gys[0] };
	}
};

// 関数クラス（2乗）
class Square : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = xs[0];
		auto y = nc::power(*x, 2);
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x = this->inputs[0]->data;
		auto gy = gys[0];
		auto gx = 2.0 * (*x) * (*gy);
		return { as_array(gx) };
	}
};

//----------------------------------
// function
//----------------------------------
// 加算
VariablePtr add(const VariablePtrList& xs)
{
	return (*std::shared_ptr<Function>(new Add()))(xs)[0];
}

// 2乗
VariablePtr square(const VariablePtr& xs)
{
	return (*std::shared_ptr<Function>(new Square()))(xs)[0];
}

void step16()
{
	auto x = as_variable(as_array({ 2.0 }));
	auto a = square(x);
	auto y = add({ square(a), square(a) });
	y->backward();

	std::cout << NdArrayPrinter(*y->data) << std::endl;
	std::cout << NdArrayPrinter(*x->grad) << std::endl;
}

}
