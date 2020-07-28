
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step43 {

void step43()
{
	// �g�C�E�f�[�^�Z�b�g
	nc::random::seed(0);
	auto x_tmp = nc::random::rand<data_t>({ 100, 1 });
	auto y_tmp = nc::sin(2.0 * M_PI * x_tmp) + nc::random::rand<data_t>({ 100, 1 });
	auto x = as_variable(as_array(x_tmp));
	auto y = as_variable(as_array(y_tmp));

	// �@�d�݂̏�����
	uint32_t I = 1;		// ���͑w�̐�
	uint32_t H = 10;	// �B��w�̐�
	uint32_t O = 1;		// �o�͑w�̐�
	auto W1 = as_variable(as_array(0.01 * nc::random::randN<data_t>({ I, H })));
	auto b1 = as_variable(as_array(nc::zeros<data_t>({ 1, H })));
	auto W2 = as_variable(as_array(0.01 * nc::random::randN<data_t>({ H, O })));
	auto b2 = as_variable(as_array(nc::zeros<data_t>({ 1, O })));

	// �A�j���[�����l�b�g���[�N�̐��_
	auto predict = [&](const VariablePtr& x) {
		auto y = linear(x, W1, b1);
		y = sigmoid(y);
		y = linear(y, W2, b2);
		return y;
	};
	auto predict_simple = [&](const VariablePtr& x) {
		auto y = linear_simple(x, W1, b1);
		y = sigmoid_simple(y);
		y = linear_simple(y, W2, b2);
		return y;
	};

	auto lr = 0.2;
	auto iters = 10000;

	// �B�j���[�����l�b�g���[�N�̊w�K
	for (int i = 0; i < iters; i++) {
		auto y_pred = predict(x);
		//auto y_pred = predict_simple(x);
		auto loss = mean_squared_error(y, y_pred);

		W1->cleargrad();
		b1->cleargrad();
		W2->cleargrad();
		b2->cleargrad();
		loss->backward();

		*(W1->data) -= lr * *(W1->grad->data);
		*(b1->data) -= lr * *(b1->grad->data);
		*(W2->data) -= lr * *(W2->grad->data);
		*(b2->data) -= lr * *(b2->grad->data);
		// 1000�񂲂Ƃɏo��
		if (i % 1000 == 0) {
			std::cout << loss << std::endl;
		}
	}
}

}
