#pragma once

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <list>
#include <vector>
#include <set>
#include <map>
#include "NumCpp.hpp"

namespace dz
{

// クラス前方宣言
class Variable;
class Function;

//----------------------------------
// type
//----------------------------------

// 基本データ型
using data_t = double;	// TODO: 最終的には float にする
using NdArray = nc::NdArray<data_t>;

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

// NdArrayPtr生成関数
extern NdArrayPtr as_array(nullptr_t = nullptr);
extern NdArrayPtr as_array(std::initializer_list<NdArray::value_type> list);
extern NdArrayPtr as_array(NdArray::value_type scalar);
extern NdArrayPtr as_array(const NdArray& data);

// VariablePtr生成関数
extern VariablePtr as_variable(nullptr_t = nullptr);
extern VariablePtr as_variable(const NdArrayPtr& data);
extern VariablePtr as_variable(const Variable& data);

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
	std::map<std::string, bool> param;

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

// NdArrayの出力ヘルパークラス
class NdArrayPrinter
{
public:
	const std::shared_ptr<NdArray> data;
	NdArrayPrinter(const std::shared_ptr<NdArray>& data) :
		data(data)
	{}
	NdArrayPrinter(const NdArray& data) :
		data(std::make_shared<NdArray>(data))
	{}
};

extern std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda);

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

extern std::ostream& operator<<(std::ostream& ost, const Variable& v);
extern std::ostream& operator<<(std::ostream& ost, const VariablePtr& p);

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
extern VariablePtr add(const VariablePtr& x0, const VariablePtr& x1);
// 減算
extern VariablePtr sub(const VariablePtr& x0, const VariablePtr& x1);
// 乗算
extern VariablePtr mul(const VariablePtr& x0, const VariablePtr& x1);
// 除算
extern VariablePtr div(const VariablePtr& x0, const VariablePtr& x1);
// 正数
extern VariablePtr pos(const VariablePtr& x);
// 負数
extern VariablePtr neg(const VariablePtr& x);
// 累乗
extern VariablePtr pow(const VariablePtr& x0, uint32_t c);
// 2乗
extern VariablePtr square(const VariablePtr& x0);

// VariablePtrの演算子オーバーロード
// 二項演算子 +
extern VariablePtr operator+(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator+(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator+(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator+(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator+(data_t lhs, const VariablePtr& rhs);
// 二項演算子 -
extern VariablePtr operator-(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator-(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator-(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator-(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator-(data_t lhs, const VariablePtr& rhs);
// 二項演算子 *
extern VariablePtr operator*(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator*(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator*(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator*(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator*(data_t lhs, const VariablePtr& rhs);
// 二項演算子 /
extern VariablePtr operator/(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator/(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator/(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator/(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator/(data_t lhs, const VariablePtr& rhs);
// 単項演算子 +
extern VariablePtr operator+(const VariablePtr& data);
// 単項演算子 -
extern VariablePtr operator-(const VariablePtr& data);

}	// namespace dezerocpp
