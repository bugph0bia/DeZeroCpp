#pragma once

#include "../dezero/dezero.hpp"

namespace F = dz::functions;

namespace dz::layers
{

// クラス前方宣言
class Layer;

//----------------------------------
// type
//----------------------------------

// スマートポインタ型
using LayerPtr = std::shared_ptr<Layer>;

// プロパティの値
using prop_value_t = std::variant<VariablePtr, LayerPtr>;
// プロパティコレクション
using props_t = std::unordered_map<std::string, prop_value_t>;
// パラメータコレクション
using params_t = std::set<VariablePtr>;
// パラメータ名称コレクション
using param_names_t = std::set<std::string>;

//----------------------------------
// class
//----------------------------------

// プロパティのset/getの代理処理クラス
class PropProxy
{
private:
	// プロパティ
	props_t& props;
	// プロパティ内のパラメータ一覧
	param_names_t& param_names;
	// キー
	const std::string& key;

public:
	// コンストラクタ
	PropProxy(props_t& props, param_names_t& param_names, const std::string& key) :
		props(props),
		param_names(param_names),
		key(key)
	{}

	// コピー/ムーブコンストラクタ
	PropProxy(const PropProxy&) = default;
	PropProxy(PropProxy&&) = default;

	// VariablePtrのコピー代入演算子 (set代理)
	PropProxy& operator=(const VariablePtr& value)
	{
		return this->set<VariablePtr>(value);
	}

	// LayerPtrのコピー代入演算子 (set代理)
	PropProxy& operator=(const LayerPtr& value)
	{
		return this->set<LayerPtr>(value);
	}

	// VariablePtrへのキャスト演算子 (get代理)
	operator VariablePtr() const noexcept
	{
		return this->get<VariablePtr>();
	}

	// LayerPtrへのキャスト演算子 (get代理)
	operator LayerPtr() const noexcept
	{
		return this->get<LayerPtr>();
	}

	// アロー演算子で対象の VariablePtr のメンバを直接操作
	VariablePtr operator->() const noexcept
	{
		return static_cast<VariablePtr>(*this);
	}

private:
	// set処理
	template<typename T>
	PropProxy& set(const T& value)
	{
		// ParameterPtr, LayerPtr(派生クラス含む) であれば、パラメータとして登録する
		if (typeid(*value) == typeid(Parameter)) {
			param_names.insert(key);
		}
		else {
			try {
				// Layer&にキャスト可能なら、Layerの派生クラス型である
				dynamic_cast<Layer&>(*value);
				param_names.insert(key);
			}
			catch (std::bad_cast&) {}
		}
		// プロパティへ登録
		props[key] = value;
		return *this;
	}

	// get処理
	template<typename T>
	T get() const noexcept
	{
		auto p = T();
		if (props.find(key) != props.end()) {
			if (std::holds_alternative<T>(props[key])) {
				p = std::get<T>(props[key]);
			}
			else {
				// 異なる型として参照
				assert(false);
			}
		}
		else {
			// 存在しないプロパティの参照
			assert(false);
		}

		return p;
	}

};

// レイヤクラス
class Layer
{
protected:
	// プロパティ
	props_t props;
	// プロパティ内のパラメータ一覧
	param_names_t param_names;

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

	// Layerプロパティをget
	Layer& layer(const std::string& key)
	{
		return *(static_cast<LayerPtr>((*this)[key]));
	}

	//// Variableプロパティをget
	//Variable& variable(const std::string& key)
	//{
	//	return *(static_cast<VariablePtr>((*this)[key]));
	//}

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
	params_t params()
	{
		// プロパティがパラメータであるか判断
		auto is_param = [this](const props_t::value_type& kv)
		{
			return this->param_names.find(kv.first) != this->param_names.end();
		};

		// プロパティからパラメータのみを抽出
		props_t props_param_only;
		std::copy_if(this->props.begin(), this->props.end(), std::inserter(props_param_only, props_param_only.end()), is_param);

		// パラメータの値のコレクションを返す
		params_t param_values;
		for (auto& kv : props_param_only) {
			// プロパティがVariablePtr
			if (std::holds_alternative<VariablePtr>(kv.second)) {
				auto v = std::get<VariablePtr>(kv.second);

				// 実態はParameterPtr
				if (typeid(*v) == typeid(Parameter)) {
					param_values.insert(v);
				}
			}
			// プロパティがLayerPtr
			if (std::holds_alternative<LayerPtr>(kv.second)) {
				auto l = std::get<LayerPtr>(kv.second);

				// 下位レイヤのパラメータを挿入
				auto params_from_layer = l->params();
				param_values.insert(params_from_layer.begin(), params_from_layer.end());
			}
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
