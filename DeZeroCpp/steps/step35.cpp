
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;
namespace F = functions;

namespace step35 {

void step35()
{
	auto x = as_variable(as_array(1.0));
	auto y = F::tanh(x);
	x->name = "x";
	y->name = "y";
	y->backward(false, true);

	int iters = 0;

	for (int i = 0; i < iters; i++) {
		auto gx = x->grad;
		x->cleargrad();
		gx->backward(false, true);
	}

	auto gx = x->grad;
	std::ostringstream osst;
	osst << iters + 1;
	gx->name = "gx" + osst.str();
	utils::plot_dot_graph(gx, false, "tanh.png");
}

}
