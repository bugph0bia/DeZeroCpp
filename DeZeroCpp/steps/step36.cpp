
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step36 {

void step36()
{
	auto x = as_variable(as_array(2.0));
	auto y = power(x, 2);
	y->backward(false, true);
	auto gx = x->grad;
	x->cleargrad();

	auto z = power(gx, 3) + y;
	z->backward();
	std::cout << x->grad << std::endl;
}

}
