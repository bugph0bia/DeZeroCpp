
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step34 {

void step34()
{
	auto x = as_variable(as_array(1.0));
	auto y = sin(x);
	y->backward(false, true);

	for (int i = 0; i < 3; i++) {

		auto gx = x->grad;
		x->cleargrad();
		gx->backward(false, true);
		std::cout << x->grad << std::endl;
	}
}

}
