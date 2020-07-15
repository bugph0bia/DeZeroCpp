
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step11 {

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
	std::vector<VariablePtr> inputs;
	// �o�̓f�[�^
	std::vector<VariablePtr> outputs;

	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	std::vector<VariablePtr> operator()(const VariablePtr& input)
	{
		return (*this)(std::vector<VariablePtr>({ input }));
	}

	// ()���Z�q
	std::vector<VariablePtr> operator()(const std::vector<VariablePtr>& inputs)
	{
		auto xs = std::vector<NdArrayPtr>();
		for(const auto& i : inputs) {
			xs.push_back(i->data);
		}

		auto ys = this->forward(xs);
		auto outputs = std::vector<VariablePtr>();
		for(const auto& y : ys) {
			auto o = as_variable(as_array(*y));
			o->set_creator(shared_from_this());
			outputs.push_back(o);
		}

		this->inputs = std::move(inputs);
		this->outputs = std::move(outputs);
		return this->outputs;
	}

	// ���`�d
	virtual std::vector<NdArrayPtr> forward(const std::vector<NdArrayPtr>& xs) = 0;
	// �t�`�d
	virtual NdArrayPtr backward(const NdArrayPtr& gy) = 0;
};

// �t�`�d
// ������ Function �N���X�̃����o���Q�Ƃ��Ă��邽�߂��̈ʒu�Œ�`����K�v������
void Variable::backward()
{
	//if (!this->grad) {
	//	// ���z�̏����l(1)��ݒ�
	//	auto g = nc::ones_like<data_t>(*this->data);
	//	this->grad = as_array(g);
	//}

	//// �֐����X�g
	//auto funcs = std::vector<FunctionPtr>({ this->creator });
	//while (!funcs.empty()) {
	//	// ���X�g����֐������o��
	//	auto f = funcs.back();
	//	funcs.pop_back();
	//	// �֐��̓��o�͂��擾
	//	auto x = f->input;
	//	auto y = f->output;
	//	// �t�`�d���Ă�
	//	x->grad = f->backward(y->grad);

	//	if (x->creator) {
	//		// �P�O�̊֐������X�g�ɒǉ�
	//		funcs.push_back(x->creator);
	//	}
	//}
}

// �֐��N���X�i���Z�j
class Add : public Function
{
public:
	// ���`�d
	std::vector<NdArrayPtr> forward(const std::vector<NdArrayPtr>& xs) override
	{
		auto x0 = xs[0];
		auto x1 = xs[1];
		auto y = (*x0) + (*x1);
		return std::vector<NdArrayPtr>({ as_array(y) });
	}
	// �t�`�d
	NdArrayPtr backward(const NdArrayPtr& gy) override
	{
		// �b��
		return gy;
	}
};

//----------------------------------
// function
//----------------------------------

void step11()
{
	auto xs = std::vector<VariablePtr>{ as_variable(as_array({ 2.0 })), as_variable(as_array({ 3.0 })) };
	auto f = std::shared_ptr<Function>(new Add());
	auto ys = (*f)(xs);
	auto y = ys[0];
	std::cout << NdArrayPrinter(*y->data) << std::endl;
}

}
