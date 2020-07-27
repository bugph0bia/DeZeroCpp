
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step40 {

void step40()
{
	{
		auto x = as_array({ 1, 2, 3 });
		auto y = broadcast_to(*x, { 2, 3 });
		std::cout << NdArrayPrinter(y) << std::endl;
		std::cout << std::endl;
	}
	{
		auto x = as_array({ {1, 2, 3}, {4, 5, 6} });
		auto y = sum_to(*x, { 1, 3 });
		std::cout << NdArrayPrinter(y) << std::endl;

		y = sum_to(*x, { 2, 1 });
		std::cout << NdArrayPrinter(y) << std::endl;
		std::cout << std::endl;
	}
	{
		auto x0 = as_array({ 1, 2, 3 });
		auto x1 = as_array(10);

		// NdArrayは自動ブロードキャストが行われないので手動で実行
		broadcast_mutual(*x0, *x1);

		auto y = *x0 + *x1;
		std::cout << NdArrayPrinter(y) << std::endl;
		std::cout << std::endl;
	}
	{
		auto x0 = as_variable(as_array({ 1, 2, 3 }));
		auto x1 = as_variable(as_array(10));

		auto y = x0 + x1;
		std::cout << y << std::endl;

		y->backward();
		std::cout << x1->grad << std::endl;
		std::cout << std::endl;
	}
}

}
