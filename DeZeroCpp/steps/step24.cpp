#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step24 {

VariablePtr sphere(const VariablePtr& x, const VariablePtr& y)
{
	auto z = pow(x, 2) + pow(y, 2);
	return z;
}

VariablePtr matyas(const VariablePtr& x, const VariablePtr& y)
{
	auto z = 0.26 * (pow(x, 2) + pow(y, 2)) - 0.48 * x * y;
	return z;
}

VariablePtr goldstein(const VariablePtr& x, const VariablePtr& y)
{
	auto z = 
		(1 + pow((x + y + 1), 2) * (19 - 14 * x + 3 * pow(x, 2) - 14 * y + 6 * x * y + 3 * pow(y, 2))) *
		(30 + pow((2 * x - 3 * y), 2) * (18 - 32 * x + 12 * pow(x, 2) + 48 * y - 36 * x * y + 27 * pow(y, 2)));
	return z;
}

void step24()
{
	{
		auto x = as_variable(as_array(1.0));
		auto y = as_variable(as_array(1.0));
		auto z = sphere(x, y);
		z->backward();

		std::cout << NdArrayPrinter(x->grad) << " " << NdArrayPrinter(y->grad) << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(1.0));
		auto y = as_variable(as_array(1.0));
		auto z = matyas(x, y);
		z->backward();

		std::cout << NdArrayPrinter(x->grad) << " " << NdArrayPrinter(y->grad) << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(1.0));
		auto y = as_variable(as_array(1.0));
		auto z = goldstein(x, y);
		z->backward();

		std::cout << NdArrayPrinter(x->grad) << " " << NdArrayPrinter(y->grad) << std::endl;
		std::cout << std::endl;
	}
}

}
