
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step22 {

class Variable;
class Function;

//----------------------------------
// typedef
//----------------------------------
// スマートポインタ型
using NdArrayPtr = std::shared_ptr<NdArray>;	// インスタンス生成時は std::make_shared<NdArray> 関数を使うこと
using VariablePtr = std::shared_ptr<Variable>;	// インスタンス生成時は std::make_shared<Variable> 関数を使うこと
using VariableWPtr = std::weak_ptr<Variable>;
using FunctionPtr = std::shared_ptr<Function>;	// 派生クラスのインスタンス生成時は new を使うこと
												// （make_shared を使うと Function クラスがインスタンス化されてエラーとなる）
// リスト型
using NdArrayPtrList = std::vector<NdArrayPtr>;
using VariablePtrList = std::vector<VariablePtr>;
using VariableWPtrList = std::vector<VariableWPtr>;


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
VariablePtr as_variable(nullptr_t = nullptr)
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

// 設定クラス
class Config
{
private:
	// コンストラクタ
	Config() {
		// 逆伝播可否
		param["enable_backprop"] = true;
	}

public:
	// 設定値
	std::unordered_map<std::string, bool> param;

	// コピー/ムーブ不可
	Config(const Config&) = delete;
	Config(Config&&) = delete;
	Config& operator=(const Config&) = delete;
	Config& operator=(Config&&) = delete;

	// インスタンス取得
	static Config& get_instance() {
		static Config instance;
		return instance;
	}
};

// 設定一時変更クラス
class UsingConfig
{
private:
	// 変更前の値
	std::string name;
	bool old_value;

public:
	// コンストラクタ
	UsingConfig(std::string name, bool value) :
		name(name)
	{
		// 設定変更
		old_value = Config::get_instance().param[name];
		Config::get_instance().param[name] = value;
	}
	// デストラクタ
	virtual ~UsingConfig()
	{
		// 設定復元
		Config::get_instance().param[name] = old_value;
	}

	// コピー/ムーブ不可
	UsingConfig(const UsingConfig&) = delete;
	UsingConfig(UsingConfig&&) = delete;
	UsingConfig& operator=(const UsingConfig&) = delete;
	UsingConfig& operator=(UsingConfig&&) = delete;
};

// 逆伝播可否を一時的にOFF
struct no_grad : UsingConfig
{
	no_grad() : UsingConfig("enable_backprop", false) {}
};

// 変数クラス
class Variable
{
public:
	// 内部データ
	NdArrayPtr data;
	// 名称
	std::string name;
	// 勾配
	NdArrayPtr grad;
	// 生成元の関数
	FunctionPtr creator;
	// 世代
	int generation;

	// コンストラクタ
	Variable(const NdArrayPtr& data, const std::string& name = "") :
		data(data),
		name(name),
		generation(0)
	{}

	// デストラクタ
	virtual ~Variable() {}

	// 生成元の関数を設定
	void set_creator(const FunctionPtr& func);

	// 逆伝播(再帰)
	void backward(bool retain_grad = false);

	// 微分を初期化
	void cleargrad() {
		this->grad = nullptr;
	}

	// NdArrayへ委譲するメンバ
	decltype(auto) shape() { return data->shape(); }
	decltype(auto) size() { return data->size(); }
};

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
	std::string str_s = "\n";
	std::string str_r = "\n          ";
	std::string::size_type Pos(str.find(str_s));
	while (Pos != std::string::npos)
	{
		str.replace(Pos, str_s.length(), str_r);
		Pos = str.find(str_s, Pos + str_r.length());
	}

	ost << "variable(" << str << ")";
	return ost;
}

std::ostream& operator<<(std::ostream& ost, const VariablePtr& p)
{
	if (!p) ost << "variable(Null)";
	else ost << *p;
	return ost;
}

// 関数クラス
class Function : public std::enable_shared_from_this<Function>
{
public:
	// 入力データ
	VariablePtrList inputs;
	// 出力データ
	VariableWPtrList outputs;
	// 世代
	int generation;

	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	VariablePtrList operator()(const NdArrayPtr& input)
	{
		// VariantPtrに変換して処理
		return (*this)(as_variable(input));
	}

	// ()演算子
	VariablePtrList operator()(const VariablePtr& input)
	{
		// リストに変換して処理
		return (*this)(VariablePtrList({ input }));
	}

	// ()演算子
	VariablePtrList operator()(const VariablePtrList& inputs)
	{
		// 入力データからNdArrayを取り出す
		auto xs = NdArrayPtrList();
		for (const auto& i : inputs) {
			xs.push_back(i->data);
		}

		// 順伝播
		auto ys = this->forward(xs);

		// 計算結果から出力データを作成
		auto outputs = VariablePtrList();
		for (const auto& y : ys) {
			auto o = as_variable(as_array(*y));
			o->set_creator(shared_from_this());
			outputs.push_back(o);
		}

		// 逆伝播可能の場合
		if (Config::get_instance().param["enable_backprop"]) {
			// 入力データのうち最大値の世代を自身の世代とする
			auto max_elem = std::max_element(
				inputs.cbegin(), inputs.cend(),
				[](VariablePtr lhs, VariablePtr rhs) { return lhs->generation < rhs->generation; }
			);
			this->generation = (*max_elem)->generation;

			// 入出力データを保持する
			this->inputs = inputs;
			this->outputs = VariableWPtrList();
			for (const auto& o : outputs) {
				VariableWPtr w = o;
				this->outputs.push_back(w);
			}
		}

		return outputs;
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

// 関数クラス（加算）
class Add : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 + x1;
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		return { gys[0], gys[0] };
	}
};

// 関数クラス（減算）
class Sub : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 - x1;
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto gy = *(gys[0]);
		return { as_array(gy), as_array(-gy) };
	}
};

// 関数クラス（乗算）
class Mul : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 * x1;
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x0 = *(this->inputs[0]->data);
		auto x1 = *(this->inputs[1]->data);
		auto gy = *(gys[0]);
		return { as_array(gy * x1), as_array(gy * x0) };
	}
};

// 関数クラス（除算）
class Div : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 / x1;
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x0 = *(this->inputs[0]->data);
		auto x1 = *(this->inputs[1]->data);
		auto gy = *(gys[0]);
		auto gx0 = as_array(gy / x1);
		auto gx1 = as_array(gy * (-x0 / nc::power(x1, 2)));
		return { gx0, gx1 };
	}
};

// 関数クラス（正数）
class Pos : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		// 入力値をそのまま返すだけだが、順伝播では新しいインスタンスにする必要がある
		auto x = *(xs[0]);
		return { as_array(x) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		return gys;
	}
};

// 関数クラス（負数）
class Neg : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		return { as_array(-x) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto gy = *(gys[0]);
		return { as_array(-gy) };
	}
};

// 関数クラス（累乗）
class Pow : public Function
{
public:
	uint32_t c;

	// コンストラクタ
	Pow(uint32_t c) : c(c) {}

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::power(x, this->c);
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x = *(this->inputs[0]->data);
		auto gy = *(gys[0]);
		auto c = this->c;
		auto gx = static_cast<data_t>(c) * nc::power(x, c - 1)  * gy;
		return { as_array(gx) };
	}
};

// 関数クラス（2乗）
class Square : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::power(x, 2);
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x = *(this->inputs[0]->data);
		auto gy = *(gys[0]);
		auto gx = 2.0 * x * gy;
		return { as_array(gx) };
	}
};

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
VariablePtr pow(const VariablePtr& x0, uint32_t c)
{
	return (*std::shared_ptr<Function>(new Pow(c)))(x0)[0];
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


void step22()
{
	{
		auto x = as_variable(as_array(2.0));
		auto y = -(+x);
		std::cout << y << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(2.0));
		auto y1 = 2.0 - x;
		auto y2 = x - 1.0;
		std::cout << y1 << std::endl;
		std::cout << y2 << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(2.0));
		auto y = pow(x, 3);
		std::cout << y << std::endl;
		std::cout << std::endl;
	}
}

}
