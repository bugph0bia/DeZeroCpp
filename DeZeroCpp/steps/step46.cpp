
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;
using namespace dz::models;
namespace F = functions;
namespace L = layers;

namespace step46 {

void step46()
{
	{
		// データセット
		nc::random::seed(0);
		auto x = as_variable(as_array(nc::random::rand<data_t>({ 100, 1 })));
		auto y = as_variable(as_array(nc::sin(2.0 * M_PI * *(x->data)) + nc::random::rand<data_t>({ 100, 1 })));

		double lr = 0.2;
		int max_iter = 10000;
		int hidden_size = 10;

		auto model = MLP({ hidden_size, 1 });

		//auto optimizer = optimizers::SGD(lr);
		auto optimizer = optimizers::MomentumSGD(lr);

		//optimizer.setup(std::make_shared<MLP>(model));	// 使用側のコードにスマートポインタを意識させないようにする
		optimizer.setup(&model);

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
