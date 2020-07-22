#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step27 {

// ŠÖ”ƒNƒ‰ƒX(sin)
class Sin : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::sin(x);
		return { as_array(y) };
	}
#ifdef IS_SIMPLE_CORE
	// ‹t“`”d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto gy = *(gys[0]);
		auto x = *(this->inputs[0]->data);
		auto gx = gy * nc::cos(x);
		return { as_array(gx) };
	}
#else
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto x = this->inputs[0];
		auto gx = *gy->data * nc::cos(*x->data);
		return { as_variable(as_array(gx)) };
	}
#endif	// #ifdef IS_SIMPLE_CORE
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

// ”÷•ª’l‚ª‚¤‚Ü‚­o‚È‚¢
//VariablePtr my_sin(const VariablePtr& x, double threshold = 0.0001)
//{
//	// ‘æ1€–Úii=0j‚Ì’l
//	auto term = x;
//	auto y = term;
//	// ‘æ2€–ÚˆÈ~
//	for (int i = 1; i < 100000; i++) {
//		term = as_variable(as_array(*term->data));
//
//		// ŠKæ‚ðŽg‚Á‚Ä‚µ‚Ü‚¤‚ÆƒI[ƒo[ƒtƒ[‚·‚é‚Ì‚Å
//		// ‘ã‚í‚è‚É1‚Â‘O‚Ì€‚Ì’l‚ð—˜—p‚µ‚ÄŒvŽZ
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
#ifdef IS_SIMPLE_CORE
		std::cout << NdArrayPrinter(x->grad) << std::endl;
#else
		std::cout << x->grad << std::endl;
#endif	// #ifdef IS_SIMPLE_CORE
		std::cout << std::endl;
	}
	{
		auto x = as_variable(as_array(M_PI / 4));
		auto y = my_sin(x, 1e-150);
		y->backward();

		std::cout << NdArrayPrinter(y->data) << std::endl;
#ifdef IS_SIMPLE_CORE
		std::cout << NdArrayPrinter(x->grad) << std::endl;
#else
		std::cout << x->grad << std::endl;
#endif	// #ifdef IS_SIMPLE_CORE
		std::cout << std::endl;

		x->name = "x";
		y->name = "y";
		plot_dot_graph(y, false);
	}
}

}
