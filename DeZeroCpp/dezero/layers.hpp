#pragma once

#include "../dezero/dezero.hpp"

namespace dz::layers
{

namespace F = functions;

//----------------------------------
// class
//----------------------------------

// プロパティのset/getの代理処理クラス
class PropProxy
{
private:
	// プロパティ
	std::unordered_map<std::string, VariablePtr>& props;
	// プロパティ内のパラメータ一覧
	std::set<std::string>& param_names;
	// キー
	const std::string& key;

public:
	// コンストラクタ
	PropProxy(std::unordered_map<std::string, VariablePtr>& props, std::set<std::string>& param_names, const std::string& key) :
		props(props),
		param_names(param_names),
		key(key)
	{}

	// コピー/ムーブコンストラクタ
	PropProxy(const PropProxy&) = default;
	PropProxy(PropProxy&&) = default;

	// コピー代入演算子 (set代理)
	PropProxy& operator=(const VariablePtr& value)
	{
		if (typeid(*value) == typeid(Parameter)) {
			param_names.insert(key);
		}
		props[key] = value;
		return *this;
	}

	// VariablePtrへのキャスト演算子 (get代理)
	operator VariablePtr() const noexcept
	{
		auto v = VariablePtr();
		if (props.find(key) != props.end())
			v = props[key];
		else
			assert(false);	// 存在しないプロパティの参照

		return v;
	}

	// アロー演算子で対象の VariablePtr のメンバを直接操作
	VariablePtr operator->() const noexcept
	{
		return static_cast<VariablePtr>(*this);
	}
};

class Layer
{
protected:
	using props_type = std::unordered_map<std::string, VariablePtr>;
	using params_type = std::set<std::string>;

	// プロパティ
	props_type props;
	// プロパティ内のパラメータ一覧
	params_type param_names;

	// 入力データ
	VariableWPtrList inputs;
	// 出力データ
	VariableWPtrList outputs;

public:
	// デストラクタ
	virtual ~Layer() {}

	// プロパティのset/get
	// []演算子だと this と併用するときに煩雑になるのでこちらを使用すると良い
	PropProxy prop(const std::string& key)
	{
		return (*this)[key];
	}

	// []演算子：プロパティのset/get
	PropProxy operator[](const std::string& key)
	{
		return PropProxy(props, param_names, key);
	}

	// ()演算子
	VariablePtrList operator()(const NdArrayPtr& input)
	{
		// VariantPtrに変換して処理
		return (*this)(as_variable(input));
	}
	VariablePtrList operator()(const VariablePtr& input)
	{
		// リストに変換して処理
		return (*this)(VariablePtrList({ input }));
	}
	VariablePtrList operator()(const VariablePtrList& inputs)
	{
		// 順伝播
		auto outputs = this->forward(inputs);

		// 入出力データを保持する
		this->inputs = VariableWPtrList();
		for (const auto& i : inputs) {
			VariableWPtr w = i;
			this->inputs.push_back(w);
		}
		this->outputs = VariableWPtrList();
		for (const auto& o : outputs) {
			VariableWPtr w = o;
			this->outputs.push_back(w);
		}

		return outputs;
	}

	// 順伝播
	virtual VariablePtrList forward(const VariablePtrList& xs) = 0;

	// パラメータのコレクションを生成
	decltype(auto) params()
	{
		// プロパティがパラメータであるか判断
		auto is_param = [this](const props_type::value_type& kv)
		{
			return this->param_names.find(kv.first) != this->param_names.end();
		};

		// プロパティからパラメータのみを抽出
		props_type props_param_only;
		std::copy_if(this->props.begin(), this->props.end(), std::inserter(props_param_only, props_param_only.end()), is_param);

		// プロパティの値のコレクションを返す
		std::set<VariablePtr> param_values;
		for (auto& kv : props_param_only) {
			param_values.insert(kv.second);
		}
		return param_values;
	}

	// 全パラメータの勾配を初期化
	void cleargrads()
	{
		// パラメータを抽出して勾配を初期化
		for (auto& p : this->params()) {
			p->cleargrad();
		}
	}
};

// レイヤクラス（線形変換/全結合）
class Linear : public Layer
{
public:
	// 入出力データサイズ
	uint32_t in_size;
	uint32_t out_size;

	// コンストラクタ
	Linear(uint32_t out_size, uint32_t in_size = 0, bool nobias = false) :
		in_size(in_size),
		out_size(out_size)
	{
		// 重みの初期化
		this->prop("W") = as_parameter(nullptr, "W");
		// in_size が指定されていない場合は後回し
		if (this->in_size != 0) {
			this->init_W();
		}

		// バイアスの初期化
		if (!nobias) {
			this->prop("b") = as_parameter(as_array(nc::zeros<data_t>({ 1, out_size })), "b");
		}
	}

	// 重みの初期化
	void init_W()
	{
		auto I = this->in_size;
		auto O = this->out_size;
		auto W_data = nc::random::randN<data_t>({ I, O }) * nc::sqrt<data_t>(1.0 / I);
		this->prop("W")->data = as_array(W_data);
	}

	// 順伝播
	virtual VariablePtrList forward(const VariablePtrList& xs) override
	{
		auto x = xs[0];

		// データを流すタイミングで重みを初期化
		if (!this->prop("W")->data) {
			this->in_size = x->shape().cols;
			this->init_W();
		}
		auto y = F::linear(x, this->prop("W"), this->prop("b"));
		return { y };
	}
};

}	// namespace dz::layers
