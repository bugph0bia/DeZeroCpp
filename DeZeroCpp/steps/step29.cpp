#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step29 {

VariablePtr f(const VariablePtr& x)
{
	auto y = power(x, 4) - 2 * power(x, 2);
	return y;
}

NdArrayPtr gx2(const NdArrayPtr& x)
{
	auto y = 12.0 * nc::power(*x, 2) - 4.0;
	return as_array(y);
}

void step29()
{
	auto x = as_variable(as_array(2.0));
	// ŒJ‚è•Ô‚µ‰ñ”
	int iters = 10;

	for (int i = 0; i < iters; i++) {
		std::cout << i << " " << x << std::endl;

		auto y = f(x);
		x->cleargrad();
		y->backward();

		*(x->data) -= *(x->grad) / *(gx2(x->data));
	}
}

}
