
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step42 {

VariablePtr predict(const VariablePtr& x, const VariablePtr& W, const VariablePtr& b)
{
	auto y = matmul(x, W) + b;
	return y;
}

VariablePtr mean_squared_error(const VariablePtr& x0, const VariablePtr& x1)
{
	auto diff = x0 - x1;
	return sum(power(diff, 2)) / diff->size();
}

void step42()
{
	// トイ・データセット
	nc::random::seed(0);
	auto x_tmp = nc::random::rand<data_t>({ 100, 1 });
	auto y_tmp = 5.0 + 2.0 * x_tmp + nc::random::rand<data_t>({ 100, 1 });
	auto x = as_variable(as_array(x_tmp));
	auto y = as_variable(as_array(y_tmp));

	auto W = as_variable(as_array(nc::zeros<data_t>({ 1, 1 })));
	auto b = as_variable(as_array(nc::zeros<data_t>({ 1, 1 })));

	double lr = 0.1;
	int iters = 100;

	for (int i = 0; i < iters; i++) {
		auto y_pred = predict(x, W, b);
		//auto loss = step42::mean_squared_error(y, y_pred);
		auto loss = dz::mean_squared_error(y, y_pred);

		W->cleargrad();
		b->cleargrad();
		loss->backward();

		*(W->data) -= lr * *(W->grad->data);
		*(b->data) -= lr * *(b->grad->data);
		std::cout << W << " " << b << " " << loss << std::endl;
	}
	std::cout << std::endl;
}

}
