
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;
using namespace dz::models;
namespace F = functions;
namespace L = layers;

namespace step45 {

void step45()
{
	{
		// データセット
		nc::random::seed(0);
		auto x = as_variable(as_array(nc::random::rand<data_t>({ 100, 1 })));
		auto y = as_variable(as_array(nc::sin(2.0 * M_PI * *(x->data)) + nc::random::rand<data_t>({ 100, 1 })));

		double lr = 0.2;
		int max_iter = 10000;
		int hidden_size = 10;

		class TwoLayerNet : public Model
		{
		public:
			// コンストラクタ
			TwoLayerNet(int hidden_size, int out_size)
			{
				this->prop("l1") = std::make_shared<L::Linear>(hidden_size);
				this->prop("l2") = std::make_shared<L::Linear>(out_size);
			}

			// 順伝播
			VariablePtrList forward(const VariablePtrList& xs) override
			{
				//auto y = (*(static_cast<L::LayerPtr>(this->prop("l1"))))(x);	// layer関数が無いと煩雑になる
				auto y = this->layer("l1")(xs);
				y = F::sigmoid(y);
				y = this->layer("l2")(y);
				return y;
			}
		};

		auto model = TwoLayerNet(hidden_size, 1);
		model.plot({ x });

		for (int i = 0; i < max_iter; i++) {
			auto y_pred = model(x)[0];
			auto loss = F::mean_squared_error(y, y_pred);

			model.cleargrads();
			loss->backward();

			for (auto& p : model.params()) {
				*(p->data) -= lr * *(p->grad->data);
			}

			// 1000回ずつ出力
			//if (i % 1000 == 0) {
			if (i % 100 == 0) {
				std::cout << loss << std::endl;
			}
		}
		std::cout << std::endl;
	}
}

}
