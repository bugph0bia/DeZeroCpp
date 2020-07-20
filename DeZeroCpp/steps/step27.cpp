#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step27 {

// �֐��N���X(sin)
class Sin : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::sin(x);
		return { as_array(y) };
	}
	// �t�`�d
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
	// ��1���ځii=0�j�̒l
	auto y = x;
	auto term = x;
	// ��2���ڈȍ~
	for (int i = 1; i < 100000; i++) {
		// �K����g���Ă��܂��ƃI�[�o�[�t���[����̂�
		// �����1�O�̍��̒l�𗘗p���Čv�Z
		auto t = std::pow(-1, i) / ((2 * i + 1) * (2 * i));
		term = t * power(x, 2) * term;
		y = y + term;
		if (std::abs((*(term->data))[0]) < threshold) {
			break;
		}
	}

	return y;
}

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
		auto x = as_variable(as_array(M_PI / 4));
		auto y = my_sin(x, 1e-150);
		y->backward();

		std::cout << NdArrayPrinter(y->data) << std::endl;
		std::cout << NdArrayPrinter(x->grad) << std::endl;
		std::cout << std::endl;
	}
}

}
