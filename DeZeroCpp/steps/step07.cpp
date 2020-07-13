
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step07 {

//----------------------------------
// class
//----------------------------------
class Variable;
class Function;

// �ϐ��N���X
class Variable
{
//protected:	// ���̃X�e�b�v�ł͈ꎞ�I��public�ɂ���
public:
	// ���z
	NdArray* grad;
	// �������̊֐�
	Function* creator;

public:
	// �����f�[�^
	NdArray	data;

	// �R���X�g���N�^
	Variable(const NdArray& data) :
		data(data)
	{}

	// �f�X�g���N�^
	virtual ~Variable()
	{
		if (grad) delete grad;
	}

	// �������̊֐���ݒ�
	void set_creator(Function* func)
	{
		creator = func;
	}

	// �t�`�d(�ċA)
	void backward();
};

// �֐��N���X
class Function
{
//protected:	// ���̃X�e�b�v�ł͈ꎞ�I��public�ɂ���
public:
	// ���̓f�[�^
	Variable* input;
	// �o�̓f�[�^
	Variable* output;

public:
	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	Variable operator()(Variable& input)
	{
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		output.set_creator(this);
		this->input = &input;
		this->output = &output;
		return output;
	}

	// ���`�d
	virtual NdArray forward(const NdArray& x)
	{
		// No Implemented
		assert(false);
		return x;
	}
	// �t�`�d
	virtual NdArray backward(const NdArray& gy)
	{
		// No Implemented
		assert(false);
		return gy;
	}
};

// �֐��N���X�i2��j
class Square : public Function
{
public:
	// ���`�d
	NdArray forward(const NdArray& x) override
	{
		return nc::power(x, 2);
	}
	// �t�`�d
	NdArray backward(const NdArray& gy) override
	{
		auto x = this->input->data;
		auto gx = 2.0 * x * gy;
		return gx;
	}
};

// �֐��N���X�iexp�j
class Exp : public Function
{
public:
	// ���`�d
	NdArray forward(const NdArray& x) override
	{
		return nc::exp(x);
	}
	// �t�`�d
	NdArray backward(const NdArray& gy) override
	{
		auto x = this->input->data;
		auto gx = nc::exp(x) * gy;
		return gx;
	}
};

// �t�`�d(�ċA)
// ������ Function �N���X�̃����o���Q�Ƃ��Ă��邽�߂��̈ʒu�Œ�`����K�v������
void Variable::backward()
{
	auto f = this->creator;
	if (f != nullptr) {
		auto x = f->input;
		x->grad = new NdArray(f->backward(*(this->grad)));
		x->backward();
	}
}

//----------------------------------
// function
//----------------------------------

void step07()
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);

	//assert(y.creator == &C);
	//assert(y.creator->input == &b);
	//assert(y.creator->input->creator == &B);
	//assert(y.creator->input->creator->input == &a);
	//assert(y.creator->input->creator->input->creator == &A);
	//assert(y.creator->input->creator->input->creator->input == &x);

	//y.grad = new NdArray({ 1.0 });

	//auto tmp_C = y.creator;
	//auto tmp_b = tmp_C->input;
	//tmp_b->grad = new NdArray(tmp_C->backward(*y.grad));

	//auto tmp_B = tmp_b->creator;
	//auto tmp_a = tmp_B->input;
	//tmp_a->grad = new NdArray(tmp_B->backward(*tmp_b->grad));

	//auto tmp_A = tmp_a->creator;
	//auto tmp_x = tmp_A->input;
	//tmp_x->grad = new NdArray(tmp_A->backward(*tmp_a->grad));

	//std::cout << NdArrayPrinter(*tmp_x->grad) << std::endl;

	y.grad = new NdArray({ 1.0 });
	y.backward();
	std::cout << NdArrayPrinter(*x.grad) << std::endl;
}

}
