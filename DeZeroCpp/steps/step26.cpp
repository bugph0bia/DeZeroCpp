#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step26 {

VariablePtr sphere(const VariablePtr& x, const VariablePtr& y)
{
	auto z = power(x, 2) + power(y, 2);
	return z;
}

VariablePtr matyas(const VariablePtr& x, const VariablePtr& y)
{
	auto z = 0.26 * (power(x, 2) + power(y, 2)) - 0.48 * x * y;
	return z;
}

VariablePtr goldstein(const VariablePtr& x, const VariablePtr& y)
{
	auto z =
		(1 + power((x + y + 1), 2) * (19 - 14 * x + 3 * power(x, 2) - 14 * y + 6 * x * y + 3 * power(y, 2))) *
		(30 + power((2 * x - 3 * y), 2) * (18 - 32 * x + 12 * power(x, 2) + 48 * y - 36 * x * y + 27 * power(y, 2)));
	return z;
}

void step26()
{
	//{
	//	auto x = as_variable(as_array(nc::random::randN<data_t>({ 2, 3 })));
	//	x->name = "x";
	//	std::cout << dot_var(x);
	//	std::cout << dot_var(x, true);
	//	std::cout << std::endl;
	//}
	//{
	//	auto x0 = as_variable(as_array(1.0));
	//	auto x1 = as_variable(as_array(1.0));
	//	auto y = x0 + x1;
	//	std::cout << dot_func(y->creator);
	//	std::cout << std::endl;
	//}

	auto x = as_variable(as_array(1.0));
	auto y = as_variable(as_array(1.0));
	auto z = goldstein(x, y);
	z->backward();

	x->name = "x";
	y->name = "y";
	z->name = "z";
	plot_dot_graph(z, false, "goldstein.png");
}

}
