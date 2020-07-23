
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step37 {

void step37()
{
	{
		auto x = as_variable(as_array(1.0));
		auto y = sin(x);
		std::cout << y << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(NdArray({ {1, 2, 3}, {4, 5, 6} })));
		auto y = sin(x);
		std::cout << y << std::endl;
		std::cout << std::endl;
	}
	{
		// Œ»Žž“_‚Å‚Í–¢ŽÀ‘•‚Ì‚½‚ßíœ

		//auto x = as_variable(as_array(NdArray({ {1, 2, 3}, {4, 5, 6} })));
		//auto c = as_variable(as_array(NdArray({ {10, 20, 30}, {40, 50, 60} })));
		//auto t = x + c;
		//auto y = sum(t);
		//y->backward(false, true);
		//std::cout << y->grad << std::endl;
		//std::cout << t->grad << std::endl;
		//std::cout << x->grad << std::endl;
		//std::cout << c->grad << std::endl;
		//std::cout << std::endl;
	}
}

}
