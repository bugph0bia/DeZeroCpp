
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;
namespace F = functions;

namespace step41 {

void step41()
{
	{
		auto a = as_array({ 1, 2, 3 });
		auto b = as_array({ 4, 5, 6 });
		auto c = as_array(nc::dot(*a, *b));
		std::cout << NdArrayPrinter(c) << std::endl;
		std::cout << std::endl;
	}
	{
		auto a = as_array({ {1, 2}, {3, 4} });
		auto b = as_array({ {5, 6}, {7, 8} });
		auto c = as_array(nc::dot(*a, *b));
		std::cout << NdArrayPrinter(c) << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(nc::random::randN<data_t>({ 2, 3 })));
		auto W = as_variable(as_array(nc::random::randN<data_t>({ 3, 4 })));
		auto y = F::matmul(x, W);
		y->backward();

		std::ostringstream osst;

		osst << x->grad->shape();
		std::cout << utils::replace_all(osst.str(), "\n", "") << std::endl;
		osst.str("");
		osst.clear();

		osst << W->grad->shape();
		std::cout << utils::replace_all(osst.str(), "\n", "") << std::endl;
		osst.str("");
		osst.clear();

		std::cout << std::endl;
	}
}

}
