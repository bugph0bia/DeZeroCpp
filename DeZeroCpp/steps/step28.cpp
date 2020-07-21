#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step28 {

VariablePtr rosenbrock(const VariablePtr& x0, const VariablePtr& x1)
{
	auto y = 100.0 * power((x1 - power(x0, 2)), 2) + power((x0 - 1.0), 2);
	return y;
}

void step28()
{
	auto x0 = as_variable(as_array(0.0));
	auto x1 = as_variable(as_array(2.0));

	// ŠwK—¦
	double lr = 0.001;
	// ŒJ‚è•Ô‚µ‰ñ”
	int iters = 1000;

	for (int i = 0; i < iters; i++) {
		std::cout << x0 << " " << x1 << std::endl;

		auto y = rosenbrock(x0, x1);

		x0->cleargrad();
		x1->cleargrad();
		y->backward();

		*(x0->data) -= lr * *(x0->grad);
		*(x1->data) -= lr * *(x1->grad);
	}
}

}
