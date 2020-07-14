
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step08 {

//----------------------------------
// class
//----------------------------------
class Variable;
class Function;

using NdArrayPtr = std::shared_ptr<NdArray>;
using VariablePtr = std::shared_ptr<Variable>;
using FunctionPtr = std::shared_ptr<Function>;


// �ϐ��N���X
class Variable
{
	//protected:	// ���̃X�e�b�v�ł͈ꎞ�I��public�ɂ���
public:
	// ���z
	NdArrayPtr grad;
	// �������̊֐�
	FunctionPtr creator;

public:
	// �����f�[�^
	NdArray	data;

	// �R���X�g���N�^
	Variable(const NdArray& data) :
		data(data)
	{}

	// �f�X�g���N�^
	virtual ~Variable() {}

	// �������̊֐���ݒ�
	void set_creator(const FunctionPtr& func)
	{
		creator = func;
	}

	// �t�`�d(�ċA)
	void backward();
};

// �֐��N���X
class Function : public std::enable_shared_from_this<Function>
{
	//protected:	// ���̃X�e�b�v�ł͈ꎞ�I��public�ɂ���
public:
	// ���̓f�[�^
	VariablePtr input;
	// �o�̓f�[�^
	VariablePtr output;

public:
	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	VariablePtr operator()(const VariablePtr& input)
	{
		auto x = input->data;
		auto y = this->forward(x);
		auto output = std::make_shared<Variable>(y);
		output->set_creator(shared_from_this());
		this->input = input;
		this->output = output;
		return output;
	}

	// ���`�d
	virtual NdArray forward(const NdArray& x) = 0;
	// �t�`�d
	virtual NdArray backward(const NdArray& gy) = 0;
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
	// �֐����X�g
	auto funcs = std::vector<FunctionPtr>({ this->creator });
	while (!funcs.empty()) {
		// ���X�g����֐������o��
		auto f = funcs.back();
		funcs.pop_back();
		// �֐��̓��o�͂��擾
		auto x = f->input;
		auto y = f->output;
		// �t�`�d���Ă�
		x->grad = std::make_shared<NdArray>(f->backward(*y->grad));

		if (x->creator != nullptr) {
			// �P�O�̊֐������X�g�ɒǉ�
			funcs.push_back(x->creator);
		}
	}
}

//----------------------------------
// function
//----------------------------------

void step08()
{
	auto A = FunctionPtr(new Square());
	auto B = FunctionPtr(new Exp());
	auto C = FunctionPtr(new Square());

	auto x = std::make_shared<Variable>(NdArray({ 0.5 }));
	auto a = (*A)(x);
	auto b = (*B)(a);
	auto y = (*C)(b);

	y->grad = std::make_shared<NdArray>(NdArray({ 1.0 }));
	y->backward();
	std::cout << NdArrayPrinter(*x->grad) << std::endl;
}

}
