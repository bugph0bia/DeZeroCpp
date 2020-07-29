
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;
namespace F = functions;

namespace step39 {

void step39()
{
	{
		auto x = as_variable(as_array({ 1, 2, 3, 4, 5, 6 }));
		auto y = F::sum(x);
		y->backward();
		std::cout << y << std::endl;
		std::cout << x->grad << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array({ {1, 2, 3}, {4, 5, 6} }));
		auto y = F::sum(x);
		y->backward();
		std::cout << y << std::endl;
		std::cout << x->grad << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array({ {1, 2, 3}, {4, 5, 6} }));
		auto y = F::sum(x, nc::Axis::ROW);
		y->backward();
		std::cout << y << std::endl;
		std::cout << x->grad << std::endl;

		std::ostringstream osst;
		osst << x->shape() << " -> " << y->shape();
		std::cout << utils::replace_all(osst.str(), "\n", "") << std::endl;
		std::cout << std::endl;
	}
}

}
