#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step27 {

// 関数クラス(sin)
class Sin : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::sin(x);
		return { as_array(y) };
	}
	// 逆伝播
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto gy = *(gys[0]);
		auto x = *(this->inputs[0]->data);
		auto gx = gy * nc::cos(x);
		return { as_array(gx) };
	}
};

VariablePtr sin(const VariablePtr& x)
{
	return (*std::shared_ptr<Function>(new Sin()))({ x })[0];
}

VariablePtr my_sin(const VariablePtr& x, double threshold = 0.0001)
{
	auto y = as_variable(as_array(0.0));
	for (int i = 0; i < 100000; i++) {
		auto c = std::pow(-1.0, i) / factorial(2 * i + 1);
		auto t = c * power(x, 2 * i + 1);
		y = y + t;
		if (std::abs((*(t->data))[0]) < threshold) {
			break;
		}
	}

	return y;
}

// 微分値がうまく出ない
//VariablePtr my_sin(const VariablePtr& x, double threshold = 0.0001)
//{
//	// 第1項目（i=0）の値
//	auto term = x;
//	auto y = term;
//	// 第2項目以降
//	for (int i = 1; i < 100000; i++) {
//		term = as_variable(as_array(*term->data));
//
//		// 階乗を使ってしまうとオーバーフローするので
//		// 代わりに1つ前の項の値を利用して計算
//		auto t = -1.0 / ((2 * i + 1) * (2 * i));
//		term = term * power(x, 2) * t;
//		y = y + term;
//		if (std::abs((*(term->data))[0]) < threshold) {
//			break;
//		}
//	}
//
//	return y;
//}

void step27()
{
	{
		auto x = as_variable(as_array(M_PI / 4));
		auto y = sin(x);
		y->backward();

		std::cout << NdArrayPrinter(y->data) << std::endl;
		std::cout << NdArrayPrinter(x->grad) << std::endl;
		std::cout << std::endl;
	}
	{
		//auto y2 = my_sin2(M_PI / 4);

		auto x = as_variable(as_array(M_PI / 4));
		auto y = my_sin(x);
		y->backward();

		std::cout << NdArrayPrinter(y->data) << std::endl;
		std::cout << NdArrayPrinter(x->grad) << std::endl;
		std::cout << std::endl;

		x->name = "x";
		y->name = "y";
		plot_dot_graph(y, false);
	}
}

}
