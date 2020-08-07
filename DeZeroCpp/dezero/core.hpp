#pragma once

#include "../dezero/dezero.hpp"

namespace dz
{

// クラス前方宣言
class Variable;
class Parameter;
class Function;

//----------------------------------
// type
//----------------------------------

// 基本データ型
using data_t = double;	// TODO: 最終的には float にする
using NdArray = nc::NdArray<data_t>;

// スマートポインタ型
using NdArrayPtr = std::shared_ptr<NdArray>;
using VariablePtr = std::shared_ptr<Variable>;
using VariableWPtr = std::weak_ptr<Variable>;
using ParameterPtr = std::shared_ptr<Parameter>;
using FunctionPtr = std::shared_ptr<Function>;

// リスト型
using NdArrayPtrList = std::vector<NdArrayPtr>;
using VariablePtrList = std::vector<VariablePtr>;
using VariableWPtrList = std::vector<VariableWPtr>;

// NdArrayPtr生成関数
inline NdArrayPtr as_array(nullptr_t /*=nullptr*/)
{
	return NdArrayPtr();	// 引数なしまたは nullptr の場合は Empty とする
}
inline NdArrayPtr as_array(std::initializer_list<NdArray::value_type> list)
{
	return std::make_shared<NdArray>(list);
}
inline NdArrayPtr as_array(NdArray::value_type scalar)
{
	return as_array({ scalar });
}
inline NdArrayPtr as_array(const NdArray& data)
{
	return std::make_shared<NdArray>(data);
}

// VariablePtr生成関数
inline VariablePtr as_variable(nullptr_t = nullptr)
{
	return VariablePtr();	// 引数なしまたは nullptr の場合は Empty とする
}
inline VariablePtr as_variable(const NdArrayPtr& data, const std::string& name = "")
{
	return std::make_shared<Variable>(data, name);
}
inline VariablePtr as_variable(const Variable& data)
{
	return std::make_shared<Variable>(data);
}

//----------------------------------
// prototype
//----------------------------------
extern inline VariablePtr add(const VariablePtr& x0, const VariablePtr& x1);
extern inline VariablePtr sub(const VariablePtr& x0, const VariablePtr& x1);
extern inline VariablePtr mul(const VariablePtr& x0, const VariablePtr& x1);
extern inline VariablePtr div(const VariablePtr& x0, const VariablePtr& x1);
extern inline VariablePtr pos(const VariablePtr& x);
extern inline VariablePtr neg(const VariablePtr& x);
extern inline VariablePtr power(const VariablePtr& x0, uint32_t c);
extern inline VariablePtr power(const NdArrayPtr& x, uint32_t c);
extern inline VariablePtr power(data_t x, uint32_t c);

extern inline VariablePtr operator+(const VariablePtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator+(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern inline VariablePtr operator+(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator+(const VariablePtr& lhs, data_t rhs);
extern inline VariablePtr operator+(data_t lhs, const VariablePtr& rhs);
extern inline VariablePtr operator-(const VariablePtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator-(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern inline VariablePtr operator-(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator-(const VariablePtr& lhs, data_t rhs);
extern inline VariablePtr operator-(data_t lhs, const VariablePtr& rhs);
extern inline VariablePtr operator*(const VariablePtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator*(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern inline VariablePtr operator*(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator*(const VariablePtr& lhs, data_t rhs);
extern inline VariablePtr operator*(data_t lhs, const VariablePtr& rhs);
extern inline VariablePtr operator/(const VariablePtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator/(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern inline VariablePtr operator/(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern inline VariablePtr operator/(const VariablePtr& lhs, data_t rhs);
extern inline VariablePtr operator/(data_t lhs, const VariablePtr& rhs);
extern inline VariablePtr operator+(const VariablePtr& data);
extern inline VariablePtr operator-(const VariablePtr& data);

namespace functions
{
extern inline VariablePtr sin(const VariablePtr& x);
extern inline VariablePtr cos(const VariablePtr& x);
extern inline VariablePtr tanh(const VariablePtr& x);
extern inline VariablePtr exp(const VariablePtr& x);
extern inline VariablePtr reshape(const VariablePtr& x, const nc::Shape& shape);
extern inline VariablePtr transpose(const VariablePtr& x);
extern inline VariablePtr sum(const VariablePtr& x, nc::Axis axis = nc::Axis::NONE);
extern inline VariablePtr broadcast_to(const VariablePtr& x, const nc::Shape& shape);
extern inline VariablePtr sum_to(const VariablePtr& x, const nc::Shape& shape);
extern inline VariablePtr matmul(const VariablePtr& x, const VariablePtr& W);
extern inline VariablePtr linear(const VariablePtr& x, const VariablePtr& W, const VariablePtr& b = nullptr);
extern inline VariablePtr linear_simple(const VariablePtr& x, const VariablePtr& W, const VariablePtr& b = nullptr);
extern inline VariablePtr sigmoid(const VariablePtr& x);
extern inline VariablePtr sigmoid_simple(const VariablePtr& x);
extern inline VariablePtr mean_squared_error(const VariablePtr& x0, const VariablePtr& x1);
extern inline VariablePtr softmax(const VariablePtr& x, nc::Axis axis = nc::Axis::ROW);
extern inline VariablePtr softmax_simple(const VariablePtr& x, nc::Axis axis = nc::Axis::ROW);

extern inline VariablePtrList sin(const VariablePtrList& xs);
extern inline VariablePtrList cos(const VariablePtrList& xs);
extern inline VariablePtrList tanh(const VariablePtrList& xs);
extern inline VariablePtrList exp(const VariablePtrList& xs);
extern inline VariablePtrList reshape(const VariablePtrList& xs, const nc::Shape& shape);
extern inline VariablePtrList transpose(const VariablePtrList& xs);
extern inline VariablePtrList sum(const VariablePtrList& xs, nc::Axis axis = nc::Axis::NONE);
extern inline VariablePtrList broadcast_to(const VariablePtrList& xs, const nc::Shape& shape);
extern inline VariablePtrList sum_to(const VariablePtrList& xs, const nc::Shape& shape);
extern inline VariablePtrList matmul(const VariablePtrList& xs);
extern inline VariablePtrList linear(const VariablePtrList& xs);
extern inline VariablePtrList linear_simple(const VariablePtrList& xs);
extern inline VariablePtrList sigmoid(const VariablePtrList& xs);
extern inline VariablePtrList sigmoid_simple(const VariablePtrList& xs);
extern inline VariablePtrList mean_squared_error(const VariablePtrList& xs);
extern inline VariablePtrList softmax(const VariablePtrList& xs, nc::Axis axis = nc::Axis::ROW);
extern inline VariablePtrList softmax_simple(const VariablePtrList& xs, nc::Axis axis = nc::Axis::ROW);
}	// namespace functions

namespace utils
{
extern std::string replace_all(const std::string& target_str, const std::string& old_str, const std::string& new_str);
extern inline NdArray broadcast_to(const NdArray& in_array, const nc::Shape& shape);
extern inline NdArray sum_to(const NdArray& in_array, const nc::Shape& shape);
extern inline void broadcast_mutual(NdArray& a0, NdArray& a1);

extern inline void plot_dot_graph(const VariablePtr& output, bool verbose = true, const std::string& to_file = "graph.png");
}	// namespace utils

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

// 変数クラス
class Variable : public std::enable_shared_from_this<Variable>
{
public:
	// 内部データ
	NdArrayPtr data;
	// 名称
	std::string name;
	// 勾配
	VariablePtr grad;
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
	void backward(bool retain_grad = false, bool create_graph = false);

	// 勾配を初期化
	void cleargrad() {
		this->grad = nullptr;
	}

	// 同名の別関数へ委譲してクラスの利便性を高める
	decltype(auto) shape() { return data->shape(); }
	decltype(auto) size() { return data->size(); }
	void reshape(const nc::Shape& shape) { functions::reshape(shared_from_this(), shape); }
	decltype(auto) transpose() { return functions::transpose(shared_from_this()); }
	decltype(auto) sum(nc::Axis axis) { return functions::sum(shared_from_this(), axis); }
};

// パラメータクラス
class Parameter : public Variable
{
public:
	// コンストラクタ
	Parameter(const NdArrayPtr& data, const std::string& name = "") :
		Variable(data, name)
	{}
};

// ParameterPtr生成関数 (基底クラスのVariablePtr型として扱う)
inline VariablePtr as_parameter(const NdArrayPtr& data, const std::string& name = "")
{
	return std::make_shared<Parameter>(data, name);
}
inline VariablePtr as_parameter(const Parameter& data)
{
	return std::make_shared<Parameter>(data);
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
	int generation = 0;

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
	virtual VariablePtrList backward(const VariablePtrList& gy) = 0;
};

// 生成元の関数を設定
inline void Variable::set_creator(const FunctionPtr& func)
{
	creator = func;

	// 生成元の関数の世代を +1 して自身の世代とする
	this->generation = func->generation + 1;
}

// 逆伝播
// 内部で Function クラスのメンバを参照しているためこの位置で定義する必要がある
inline void Variable::backward(bool retain_grad /*=false*/, bool create_graph /*=false*/)
{
	// 勾配が未設定＝逆伝播の開始点
	if (!this->grad) {
		// 勾配の初期値(1)を設定
		auto g = nc::ones_like<data_t>(*this->data);
		this->grad = as_variable(as_array(g));
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
		auto gys = VariablePtrList();
		for (const auto& o : f->outputs) {
			gys.push_back(o.lock()->grad);
		}

		{
			// このスコープの中だけ設定変更
			UsingConfig with("enable_backprop", create_graph);

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
					// 新しいインスタンスを作ることが重要
					// 例えば、x->grad += gx; としてはいけない（付録A参照）
					x->grad = x->grad + gx;
				}

				// １つ前の関数をリストに追加
				if (x->creator) {
					add_func(x->creator);
				}
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
	// 入力データの形状
	nc::Shape x0_shape;
	nc::Shape x1_shape;

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);

		// 入力データの形状を保存
		x0_shape = x0.shape();
		x1_shape = x1.shape();

		// NdArrayの四則演算前のブロードキャスト
		utils::broadcast_mutual(x0, x1);

		auto y = x0 + x1;
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gx0 = gys[0];
		auto gx1 = gys[0];

		// 順伝播でブロードキャストが発生している場合は、ブロードキャストの逆伝播を行う
		if (this->x0_shape != this->x1_shape) {
			gx0 = functions::sum_to(gx0, this->x0_shape);
			gx1 = functions::sum_to(gx1, this->x1_shape);
		}
		return { gx0, gx1 };
	}
};

// 関数クラス（減算）
class Sub : public Function
{
public:
	// 入力データの形状
	nc::Shape x0_shape;
	nc::Shape x1_shape;

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);

		// 入力データの形状を保存
		x0_shape = x0.shape();
		x1_shape = x1.shape();

		// NdArrayの四則演算前のブロードキャスト
		utils::broadcast_mutual(x0, x1);

		auto y = x0 - x1;
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gx0 = gys[0];
		auto gx1 = -gys[0];

		// 順伝播でブロードキャストが発生している場合は、ブロードキャストの逆伝播を行う
		if (this->x0_shape != this->x1_shape) {
			gx0 = functions::sum_to(gx0, this->x0_shape);
			gx1 = functions::sum_to(gx1, this->x1_shape);
		}
		return { gx0, gx1 };
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

		// NdArrayの四則演算前のブロードキャスト
		utils::broadcast_mutual(x0, x1);

		auto y = x0 * x1;
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x0 = this->inputs[0];
		auto x1 = this->inputs[1];
		auto gx0 = gys[0] * x1;
		auto gx1 = gys[0] * x0;

		// 順伝播でブロードキャストが発生している場合は、ブロードキャストの逆伝播を行う
		if (x0->data->shape() != x1->data->shape()) {
			gx0 = functions::sum_to(gx0, x0->data->shape());
			gx1 = functions::sum_to(gx1, x1->data->shape());
		}
		return { gx0, gx1 };
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

		// NdArrayの四則演算前のブロードキャスト
		utils::broadcast_mutual(x0, x1);

		auto y = x0 / x1;
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x0 = this->inputs[0];
		auto x1 = this->inputs[1];
		auto gy = gys[0];
		auto gx0 = gy / x1;
		auto gx1 = gy * (-x0 / power(x1, 2));

		// 順伝播でブロードキャストが発生している場合は、ブロードキャストの逆伝播を行う
		if (x0->data->shape() != x1->data->shape()) {
			gx0 = functions::sum_to(gx0, x0->data->shape());
			gx1 = functions::sum_to(gx1, x1->data->shape());
		}
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
	VariablePtrList backward(const VariablePtrList& gys) override
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
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		return { -gy };
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
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto c = this->c;
		auto gx = static_cast<data_t>(c)* power(x, c - 1) * gy;
		return { gx };
	}
};

//----------------------------------
// function
//----------------------------------

inline std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// nullptr の場合
	if (!nda.data) ost << "Null";
	// NdArrayがスカラーなら中身のデータを標準出力へ
	else if (nda.data->shape().rows == 1 && nda.data->shape().cols == 1) ost << (*nda.data)[0];
	// 通常時
	else ost << *(nda.data);
	return ost;
}

inline std::ostream& operator<<(std::ostream& ost, const Variable& v)
{
	std::ostringstream osst;
	// 標準出力の小数点以下桁数を 15 とする
	osst << std::fixed << std::setprecision(15);
	osst << NdArrayPrinter(v.data);
	auto str = osst.str();

	// 末尾の改行を削除
	if (str.back() == '\n') str.pop_back();

	// 途中の改行にインデントを追加
	str = utils::replace_all(str, "\n", "\n          ");

	ost << "variable(" << str << ")";
	return ost;
}

inline std::ostream& operator<<(std::ostream& ost, const VariablePtr& p)
{
	if (!p) ost << "variable(Null)";
	else ost << *p;
	return ost;
}

// 加算
inline VariablePtr add(const VariablePtr& x0, const VariablePtr& x1)
{
	FunctionPtr f = std::make_shared<Add>();
	VariablePtrList args = { x0, x1 };
	auto ys = (*f)(args);
	return ys[0];
}

// 減算
inline VariablePtr sub(const VariablePtr& x0, const VariablePtr& x1)
{
	FunctionPtr f = std::make_shared<Sub>();
	VariablePtrList args = { x0, x1 };
	auto ys = (*f)(args);
	return ys[0];
}

// 乗算
inline VariablePtr mul(const VariablePtr& x0, const VariablePtr& x1)
{
	FunctionPtr f = std::make_shared<Mul>();
	VariablePtrList args = { x0, x1 };
	auto ys = (*f)(args);
	return ys[0];
}

// 除算
inline VariablePtr div(const VariablePtr& x0, const VariablePtr& x1)
{
	FunctionPtr f = std::make_shared<Div>();
	VariablePtrList args = { x0, x1 };
	auto ys = (*f)(args);
	return ys[0];
}

// 正数
inline VariablePtr pos(const VariablePtr& x)
{
	FunctionPtr f = std::make_shared<Pos>();
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// 負数
inline VariablePtr neg(const VariablePtr& x)
{
	FunctionPtr f = std::make_shared<Neg>();
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// 累乗
inline VariablePtr power(const VariablePtr& x, uint32_t c)
{
	FunctionPtr f = std::make_shared<Pow>(c);
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtr power(const NdArrayPtr& x, uint32_t c)
{
	return power(as_variable(x), c);
}
inline VariablePtr power(data_t x, uint32_t c)
{
	return power(as_variable(as_array(x)), c);
}

// VariablePtrの演算子オーバーロード
// 二項演算子 +
inline VariablePtr operator+(const VariablePtr& lhs, const VariablePtr& rhs) { return add(lhs, rhs); }
inline VariablePtr operator+(const VariablePtr& lhs, const NdArrayPtr& rhs) { return add(lhs, as_variable(rhs)); }
inline VariablePtr operator+(const NdArrayPtr& lhs, const VariablePtr& rhs) { return add(as_variable(lhs), rhs); }
inline VariablePtr operator+(const VariablePtr& lhs, data_t rhs) { return add(lhs, as_variable(as_array(rhs))); }
inline VariablePtr operator+(data_t lhs, const VariablePtr& rhs) { return add(as_variable(as_array(lhs)), rhs); }
// 二項演算子 -
inline VariablePtr operator-(const VariablePtr& lhs, const VariablePtr& rhs) { return sub(lhs, rhs); }
inline VariablePtr operator-(const VariablePtr& lhs, const NdArrayPtr& rhs) { return sub(lhs, as_variable(rhs)); }
inline VariablePtr operator-(const NdArrayPtr& lhs, const VariablePtr& rhs) { return sub(as_variable(lhs), rhs); }
inline VariablePtr operator-(const VariablePtr& lhs, data_t rhs) { return sub(lhs, as_variable(as_array(rhs))); }
inline VariablePtr operator-(data_t lhs, const VariablePtr& rhs) { return sub(as_variable(as_array(lhs)), rhs); }
// 二項演算子 *
inline VariablePtr operator*(const VariablePtr& lhs, const VariablePtr& rhs) { return mul(lhs, rhs); }
inline VariablePtr operator*(const VariablePtr& lhs, const NdArrayPtr& rhs) { return mul(lhs, as_variable(rhs)); }
inline VariablePtr operator*(const NdArrayPtr& lhs, const VariablePtr& rhs) { return mul(as_variable(lhs), rhs); }
inline VariablePtr operator*(const VariablePtr& lhs, data_t rhs) { return mul(lhs, as_variable(as_array(rhs))); }
inline VariablePtr operator*(data_t lhs, const VariablePtr& rhs) { return mul(as_variable(as_array(lhs)), rhs); }
// 二項演算子 /
inline VariablePtr operator/(const VariablePtr& lhs, const VariablePtr& rhs) { return div(lhs, rhs); }
inline VariablePtr operator/(const VariablePtr& lhs, const NdArrayPtr& rhs) { return div(lhs, as_variable(rhs)); }
inline VariablePtr operator/(const NdArrayPtr& lhs, const VariablePtr& rhs) { return div(as_variable(lhs), rhs); }
inline VariablePtr operator/(const VariablePtr& lhs, data_t rhs) { return div(lhs, as_variable(as_array(rhs))); }
inline VariablePtr operator/(data_t lhs, const VariablePtr& rhs) { return div(as_variable(as_array(lhs)), rhs); }
// 単項演算子 +
inline VariablePtr operator+(const VariablePtr& data) { return pos(data); }
// 単項演算子 -
inline VariablePtr operator-(const VariablePtr& data) { return neg(data); }

}	// namespace dezerocpp
