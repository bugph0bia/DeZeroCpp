
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;
namespace F = functions;
namespace L = layers;

namespace step44 {

void step44()
{
	{
		auto x = as_variable(as_array(1.0));
		auto p = as_parameter(as_array(2.0));
		auto y = x * p;

		std::cout << (typeid(*p) == typeid(Parameter) ? "True" : "False") << std::endl;
		std::cout << (typeid(*x) == typeid(Parameter) ? "True" : "False") << std::endl;
		std::cout << (typeid(*y) == typeid(Parameter) ? "True" : "False") << std::endl;
		std::cout << std::endl;
	}
	{
		// 一時的にしかコンパイルが通らないコード

		//auto layer = Layer();

		//layer["p1"] = as_parameter(as_array(1.0));
		//layer["p2"] = as_parameter(as_array(2.0));
		//layer["p3"] = as_variable(as_array(3.0));
		////layer["p4"] = "test";

		//std::cout << "layer.params" << std::endl;
		//std::cout << "----------------" << std::endl;

		//for (const auto name : layer.params) {
		//	std::cout << name << " " << layer.props[name] << std::endl;
		//}
		//std::cout << std::endl;
	}
	{
		// データセット
		nc::random::seed(0);
		auto x = as_variable(as_array(nc::random::rand<data_t>({ 100, 1 })));
		auto y = as_variable(as_array(nc::sin(2.0 * M_PI * *(x->data)) + nc::random::rand<data_t>({ 100, 1 })));

		auto l1 = L::Linear(10);
		auto l2 = L::Linear(1);

		auto predict = [&](const VariablePtr& x)
		{
			auto y = l1(x)[0];
			y = F::sigmoid(y);
			y = l2(y)[0];
			return y;
		};

		double lr = 0.2;
		int iters = 10000;

		for (int i = 0; i < iters; i++) {
			auto y_pred = predict(x);
			auto loss = F::mean_squared_error(y, y_pred);

			l1.cleargrads();
			l2.cleargrads();
			loss->backward();

			for (auto* pl : { &l1, &l2 }) {
				auto& l = *pl;
				for (auto& p : l.params()) {
					*(p->data) -= lr * *(p->grad->data);
				}
			}

			// 1000回ずつ出力
			if (i % 100 == 0) {
				std::cout << loss << std::endl;
			}
		}
		std::cout << std::endl;
	}
}

}
