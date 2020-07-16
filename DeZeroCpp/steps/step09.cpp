
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step09 {

class Variable;
class Function;

//----------------------------------
// typedef
//----------------------------------
// NdArray�N���X�̃X�}�[�g�|�C���^�^
// �C���X�^���X�������� std::make_shared<NdArray> �֐����g������
using NdArrayPtr = std::shared_ptr<NdArray>;

// Variable�N���X�̃X�}�[�g�|�C���^�^
// �C���X�^���X�������� std::make_shared<Variable> �֐����g������
using VariablePtr = std::shared_ptr<Variable>;

// Function�N���X�̃X�}�[�g�|�C���^�^
// �h���N���X�̃C���X�^���X�������� new ���g������
// �imake_shared ���g���� Function �N���X���C���X�^���X������ăG���[�ƂȂ�j
using FunctionPtr = std::shared_ptr<Function>;

//// std::initializer_list �� {...} �`���� std::make_shared ���邽�߂̃w���p�[�֐�
//template<typename ObjType, typename DataType>
//std::shared_ptr<ObjType> make_shared_from_list(std::initializer_list<DataType> list) {
//	return std::make_shared<ObjType>(std::move(list));
//}

// NdArrayPtr�쐬
NdArrayPtr as_array(nullptr_t = nullptr)
{
	return NdArrayPtr();	// �����Ȃ��܂��� nullptr �̏ꍇ�� Empty �Ƃ���
}
NdArrayPtr as_array(std::initializer_list<NdArray::value_type> list)
{
	return std::make_shared<NdArray>(list);
}
NdArrayPtr as_array(NdArray::value_type scalar)
{
	return as_array({ scalar });
}
NdArrayPtr as_array(const NdArray& data)
{
	return std::make_shared<NdArray>(data);
}

// VariablePtr�쐬
VariablePtr as_variable(const NdArrayPtr& data)
{
	return std::make_shared<Variable>(data);
}
VariablePtr as_variable(const Variable& data)
{
	return std::make_shared<Variable>(data);
}

//----------------------------------
// class
//----------------------------------

// �ϐ��N���X
class Variable
{
public:
	// �����f�[�^
	NdArrayPtr data;
	// ���z
	NdArrayPtr grad;
	// �������̊֐�
	FunctionPtr creator;

	// �R���X�g���N�^
	Variable(const NdArrayPtr& data) :
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
public:
	// ���̓f�[�^
	VariablePtr input;
	// �o�̓f�[�^
	VariablePtr output;

	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	VariablePtr operator()(const VariablePtr& input)
	{
		auto x = input->data;
		auto y = this->forward(x);
		auto output = as_variable(y);
		output->set_creator(shared_from_this());
		this->input = input;
		this->output = output;
		return output;
	}

	// ���`�d
	virtual NdArrayPtr forward(const NdArrayPtr& px) = 0;
	// �t�`�d
	virtual NdArrayPtr backward(const NdArrayPtr& pgy) = 0;
};

// �֐��N���X�i2��j
class Square : public Function
{
public:
	// ���`�d
	NdArrayPtr forward(const NdArrayPtr& px) override
	{
		auto x = *px;
		auto y = nc::power(x, 2);
		return as_array(y);
	}
	// �t�`�d
	NdArrayPtr backward(const NdArrayPtr& pgy) override
	{
		auto x = *this->input->data;
		auto gy = *pgy;
		auto gx = 2.0 * x * gy;
		return as_array(gx);
	}
};

// �֐��N���X�iexp�j
class Exp : public Function
{
public:
	// ���`�d
	NdArrayPtr forward(const NdArrayPtr& px) override
	{
		auto x = *px;
		auto y = nc::exp(x);
		return as_array(y);
	}
	// �t�`�d
	NdArrayPtr backward(const NdArrayPtr& pgy) override
	{
		auto x = *this->input->data;
		auto gy = *pgy;
		auto gx = nc::exp(x) * gy;
		return as_array(gx);
	}
};

// �t�`�d
// ������ Function �N���X�̃����o���Q�Ƃ��Ă��邽�߂��̈ʒu�Œ�`����K�v������
void Variable::backward()
{
	if (!this->grad) {
		// ���z�̏����l(1)��ݒ�
		auto g = nc::ones_like<data_t>(*this->data);
		this->grad = as_array(g);
	}

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
		x->grad = f->backward(y->grad);

		if (x->creator) {
			// �P�O�̊֐������X�g�ɒǉ�
			funcs.push_back(x->creator);
		}
	}
}

//----------------------------------
// function
//----------------------------------
VariablePtr square(VariablePtr x)
{
	auto f = FunctionPtr(new Square());
	return (*f)(x);
}

VariablePtr exp(VariablePtr x)
{
	auto f = FunctionPtr(new Exp());
	return (*f)(x);
}

void step09()
{
	auto x = as_variable(as_array({ 0.5 }));
	//auto a = square(x);
	//auto b = exp(a);
	//auto y = square(b);
	auto y = square(exp(square(x)));

	y->backward();
	std::cout << NdArrayPrinter(x->grad) << std::endl;
}

}
