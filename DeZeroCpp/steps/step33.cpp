
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step33 {

VariablePtr f(const VariablePtr& x)
{
	auto y = power(x, 4) - 2 * power(x, 2);
	return y;
}

void step33()
{
	{
		auto x = as_variable(as_array(2.0));
		auto y = f(x);
		y->backward(false, true);
		std::cout << x->grad << std::endl;

		auto gx = x->grad;
		x->cleargrad();
		gx->backward();
		std::cout << x->grad << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(2.0));
		int iters = 10;

		for (int i = 0; i < iters; i++) {
			std::cout << i << " " << x << std::endl;

			auto y = f(x);
			x->cleargrad();
			y->backward(false, true);

			auto gx = x->grad;
			x->cleargrad();
			gx->backward();
			auto gx2 = x->grad;

			*x->data -= *gx->data / *gx2->data;
		}
	}
}

}
