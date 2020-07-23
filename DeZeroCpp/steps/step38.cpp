
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step38 {

void step38()
{
	{
		auto x = as_variable(as_array({ {1, 2, 3}, {4, 5, 6} }));
		auto y = reshape(x, nc::Shape(6, 1));
		y->backward(false, true);
		std::cout << x->grad << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array({ {1, 2, 3}, {4, 5, 6} }));
		//auto y = transpose(x);
		auto y = x->transpose();
		y->backward(false, true);
		std::cout << x->grad << std::endl;
		std::cout << std::endl;
	}
}

}
