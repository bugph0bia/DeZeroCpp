#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step23 {

void step23()
{
	auto x = as_variable(as_array(1.0));
	auto y = power(x + 3, 2);
	y->backward();

	std::cout << y << std::endl;

#ifdef IS_SIMPLE_CORE
	std::cout << NdArrayPrinter(x->grad) << std::endl;
#else
	std::cout << x->grad << std::endl;
#endif	// #ifdef IS_SIMPLE_CORE
}

}
