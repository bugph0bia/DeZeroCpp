#pragma once

#include "../dezero/dezero.hpp"

namespace F = dz::functions;
namespace L = dz::layers;

namespace dz::models
{

//----------------------------------
// class
//----------------------------------

// モデルクラス
class Model : public layers::Layer
{
public:
	// 計算グラフをプロット
	void plot(const VariablePtrList& inputs, const std::string& to_file = "model.png") {
		auto y = this->forward(inputs);
		utils::plot_dot_graph(y[0], true, to_file);
	}
};

// 多層パーセプトロン (MLP: Multi-Layer Perceptron) クラス
class MLP : public Model
{
private:
	// レイヤーリスト
	std::vector<L::LayerPtr> layers;
	// 活性化関数
	std::function<F::function_t> activation;

public:
	// コンストラクタ
	MLP(std::vector<int> fc_output_sizes, F::function_t* activation = F::sigmoid) :
		activation(activation)
	{
		int i = 0;
		for (auto out_size : fc_output_sizes) {
			auto layer = std::make_shared<L::Linear>(out_size);
			// レイヤーをプロパティとして登録
			std::ostringstream osst;
			osst << "l" << i;
			this->prop(osst.str()) = layer;
			// 自身のレイヤーリストにも登録
			this->layers.push_back(layer);
			i++;
		}
	}

	// 順伝播
	VariablePtrList forward(const VariablePtrList& xs) override
	{
		// 最後の１つ前のレイヤまで
		auto xs_tmp = xs;
		for (auto iter = this->layers.begin(); iter != this->layers.end() - 1; iter++) {
			auto& l = **iter;
			xs_tmp = this->activation(l(xs_tmp));
		}

		// 最後のレイヤ
		auto &l = *this->layers.back();
		return l(xs_tmp);
	}
};

}	// namespace dz::models
