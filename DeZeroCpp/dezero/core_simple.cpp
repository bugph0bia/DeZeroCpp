
#include "pch.h"

#include "core_simple.hpp"

namespace dz
{

//----------------------------------
// type
//----------------------------------

// NdArrayPtr生成関数
NdArrayPtr as_array(nullptr_t /*=nullptr*/)
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

// VariablePtr生成関数
VariablePtr as_variable(nullptr_t /*=nullptr*/)
{
	return VariablePtr();	// 引数なしまたは nullptr の場合は Empty とする
}
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

std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// nullptr の場合
	if (!nda.data) ost << "Null";
	// NdArrayがスカラーなら中身のデータを標準出力へ
	else if (nda.data->shape().rows == 1 && nda.data->shape().cols == 1) ost << (*nda.data)[0];
	// 通常時
	else ost << nda.data;
	return ost;
}

std::ostream& operator<<(std::ostream& ost, const Variable& v)
{
	std::ostringstream osst;
	// 標準出力の小数点以下桁数を 15 とする
	osst << std::fixed << std::setprecision(15);
	osst << NdArrayPrinter(v.data);
	auto str = osst.str();

	// 末尾の改行を削除
	if (str.back() == '\n') str.pop_back();

	// 途中の改行にインデントを追加
	str = replace_all(str, "\n", "\n          ");

	ost << "variable(" << str << ")";
	return ost;
}

std::ostream& operator<<(std::ostream& ost, const VariablePtr& p)
{
	if (!p) ost << "variable(Null)";
	else ost << *p;
	return ost;
}

// 生成元の関数を設定
void Variable::set_creator(const FunctionPtr& func)
{
	creator = func;

	// 生成元の関数の世代を +1 して自身の世代とする
	this->generation = func->generation + 1;
}

// 逆伝播
// 内部で Function クラスのメンバを参照しているためこの位置で定義する必要がある
void Variable::backward(bool retain_grad /*=false*/)
{
	// 勾配が未設定＝逆伝播の開始点
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

	// 関数リストが空になるまでループ
	while (!funcs.empty()) {
		// リストから関数を取り出す
		auto f = funcs.back();
		funcs.pop_back();

		// 出力データから勾配を取り出す
		auto gys = NdArrayPtrList();
		for (const auto& o : f->outputs) {
			gys.push_back(o.lock()->grad);
		}

		// 逆伝播
		auto gxs = f->backward(gys);

		// 入力データと算出した勾配の要素数は一致する必要あり
		assert(f->inputs.size() == gxs.size());

		// 入力データと勾配のペアをループ
		for (size_t i = 0; i < gxs.size(); i++) {
			auto x = f->inputs[i];
			auto gx = gxs[i];

			// 勾配が未設定なら代入する
			if (!x->grad) {
				x->grad = gx;
			}
			// 勾配が設定済みなら加算する
			else {
				// 新しい NdArrayPtr インスタンスを作ることが重要
				// 例えば、*x->grad += *gx; としてはいけない（付録A参照）
				x->grad = as_array(*x->grad + *gx);
			}

			// １つ前の関数をリストに追加
			if (x->creator) {
				add_func(x->creator);
			}
		}

		// 勾配を保持しない場合
		if (!retain_grad) {
			// 勾配を削除
			for (const auto& y : f->outputs) {
				y.lock()->grad = nullptr;
			}
		}
	}
}

//----------------------------------
// function
//----------------------------------

// 加算
VariablePtr add(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Add()))({ x0, x1 })[0];
}

// 減算
VariablePtr sub(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Sub()))({ x0, x1 })[0];
}

// 乗算
VariablePtr mul(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Mul()))({ x0, x1 })[0];
}

// 除算
VariablePtr div(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Div()))({ x0, x1 })[0];
}

// 正数
VariablePtr pos(const VariablePtr& x)
{
	return (*std::shared_ptr<Function>(new Pos()))({ x })[0];
}

// 負数
VariablePtr neg(const VariablePtr& x)
{
	return (*std::shared_ptr<Function>(new Neg()))({ x })[0];
}

// 累乗
VariablePtr power(const VariablePtr& x0, uint32_t c)
{
	return (*std::shared_ptr<Function>(new Pow(c)))(x0)[0];
}
VariablePtr power(const NdArrayPtr& x, uint32_t c)
{
	return power(as_variable(x), c);
}
VariablePtr power(data_t x, uint32_t c)
{
	return power(as_variable(as_array(x)), c);
}

// 2乗
VariablePtr square(const VariablePtr& x0)
{
	return (*std::shared_ptr<Function>(new Square()))(x0)[0];
}

// VariablePtrの演算子オーバーロード
// 二項演算子 +
VariablePtr operator+(const VariablePtr& lhs, const VariablePtr& rhs) { return add(lhs, rhs); }
VariablePtr operator+(const VariablePtr& lhs, const NdArrayPtr& rhs) { return add(lhs, as_variable(rhs)); }
VariablePtr operator+(const NdArrayPtr& lhs, const VariablePtr& rhs) { return add(as_variable(lhs), rhs); }
VariablePtr operator+(const VariablePtr& lhs, data_t rhs) { return add(lhs, as_variable(as_array(rhs))); }
VariablePtr operator+(data_t lhs, const VariablePtr& rhs) { return add(as_variable(as_array(lhs)), rhs); }
// 二項演算子 -
VariablePtr operator-(const VariablePtr& lhs, const VariablePtr& rhs) { return sub(lhs, rhs); }
VariablePtr operator-(const VariablePtr& lhs, const NdArrayPtr& rhs) { return sub(lhs, as_variable(rhs)); }
VariablePtr operator-(const NdArrayPtr& lhs, const VariablePtr& rhs) { return sub(as_variable(lhs), rhs); }
VariablePtr operator-(const VariablePtr& lhs, data_t rhs) { return sub(lhs, as_variable(as_array(rhs))); }
VariablePtr operator-(data_t lhs, const VariablePtr& rhs) { return sub(as_variable(as_array(lhs)), rhs); }
// 二項演算子 *
VariablePtr operator*(const VariablePtr& lhs, const VariablePtr& rhs) { return mul(lhs, rhs); }
VariablePtr operator*(const VariablePtr& lhs, const NdArrayPtr& rhs) { return mul(lhs, as_variable(rhs)); }
VariablePtr operator*(const NdArrayPtr& lhs, const VariablePtr& rhs) { return mul(as_variable(lhs), rhs); }
VariablePtr operator*(const VariablePtr& lhs, data_t rhs) { return mul(lhs, as_variable(as_array(rhs))); }
VariablePtr operator*(data_t lhs, const VariablePtr& rhs) { return mul(as_variable(as_array(lhs)), rhs); }
// 二項演算子 /
VariablePtr operator/(const VariablePtr& lhs, const VariablePtr& rhs) { return div(lhs, rhs); }
VariablePtr operator/(const VariablePtr& lhs, const NdArrayPtr& rhs) { return div(lhs, as_variable(rhs)); }
VariablePtr operator/(const NdArrayPtr& lhs, const VariablePtr& rhs) { return div(as_variable(lhs), rhs); }
VariablePtr operator/(const VariablePtr& lhs, data_t rhs) { return div(lhs, as_variable(as_array(rhs))); }
VariablePtr operator/(data_t lhs, const VariablePtr& rhs) { return div(as_variable(as_array(lhs)), rhs); }
// 単項演算子 +
VariablePtr operator+(const VariablePtr& data) { return pos(data); }
// 単項演算子 -
VariablePtr operator-(const VariablePtr& data) { return neg(data); }

}	// namespace dezerocpp
