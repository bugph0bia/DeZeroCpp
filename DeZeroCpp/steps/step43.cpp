
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step43 {

void step43()
{
	// トイ・データセット
	nc::random::seed(0);
	auto x_tmp = nc::random::rand<data_t>({ 100, 1 });
	auto y_tmp = nc::sin(2.0 * M_PI * x_tmp) + nc::random::rand<data_t>({ 100, 1 });
	auto x = as_variable(as_array(x_tmp));
	auto y = as_variable(as_array(y_tmp));

}

}
