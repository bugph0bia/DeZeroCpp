
#include "pch.h"

#include <cassert>
#include "../dezero/dezero.hpp"

using namespace dz;

namespace step14 {

class Variable;
class Function;

//----------------------------------
// typedef
//----------------------------------
// �X�}�[�g�|�C���^�^
using NdArrayPtr = std::shared_ptr<NdArray>;	// �C���X�^���X�������� std::make_shared<NdArray> �֐����g������
using VariablePtr = std::shared_ptr<Variable>;	// �C���X�^���X�������� std::make_shared<Variable> �֐����g������
using FunctionPtr = std::shared_ptr<Function>;	// �h���N���X�̃C���X�^���X�������� new ���g������
												// �imake_shared ���g���� Function �N���X���C���X�^���X������ăG���[�ƂȂ�j
// ���X�g�^
using NdArrayPtrList = std::vector<NdArrayPtr>;
using VariablePtrList = std::vector<VariablePtr>;


//// std::initializer_list �� {...} �`���� std::make_shared ���邽�߂̃w���p�[�֐�
//template<typename ObjType, typename DataType>
//std::shared_ptr<ObjType> make_shared_from_list(std::initializer_list<DataType> list) {
//	return std::make_shared<ObjType>(std::move(list));
//}

// NdArrayPtr�쐬
// nullptr����őΉ��ł��邽�ߍ폜
//NdArrayPtr as_array(nullptr_t = nullptr)
//{
//	return NdArrayPtr();	// �����Ȃ��܂��� nullptr �̏ꍇ�� Empty �Ƃ���
//}
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

	// ������������
	void cleargrad() {
		this->grad = nullptr;
	}
};

// �֐��N���X
class Function : public std::enable_shared_from_this<Function>
{
public:
	// ���̓f�[�^
	VariablePtrList inputs;
	// �o�̓f�[�^
	VariablePtrList outputs;

	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	VariablePtrList operator()(const VariablePtr& input)
	{
		return (*this)(VariablePtrList({ input }));
	}

	// ()���Z�q
	VariablePtrList operator()(const VariablePtrList& inputs)
	{
		auto xs = NdArrayPtrList();
		for (const auto& i : inputs) {
			xs.push_back(i->data);
		}

		auto ys = this->forward(xs);
		auto outputs = VariablePtrList();
		for (const auto& y : ys) {
			auto o = as_variable(as_array(*y));
			o->set_creator(shared_from_this());
			outputs.push_back(o);
		}

		this->inputs = inputs;
		this->outputs = std::move(outputs);
		return this->outputs;
	}

	// ���`�d
	virtual NdArrayPtrList forward(const NdArrayPtrList& xs) = 0;
	// �t�`�d
	virtual NdArrayPtrList backward(const NdArrayPtrList& gy) = 0;
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
	auto funcs = std::list<FunctionPtr>({ this->creator });
	while (!funcs.empty()) {
		// ���X�g����֐������o��
		auto f = funcs.back();
		funcs.pop_back();

		auto gys = NdArrayPtrList();
		for (const auto& o : f->outputs) {
			gys.push_back(o->grad);
		}
		auto gxs = f->backward(gys);

		assert(f->inputs.size() == gxs.size());

		for (size_t i = 0; i < gxs.size(); i++) {
			auto x = f->inputs[i];
			auto gx = gxs[i];

			if (!x->grad) {
				x->grad = gx;
			}
			else {
				// �V���� NdArrayPtr �C���X�^���X����邱�Ƃ��d�v
				// �Ⴆ�΁A*x->grad += *gx; �Ƃ��Ă͂����Ȃ��i�t�^A�Q�Ɓj
				x->grad = as_array(*x->grad + *gx);
			}

			if (x->creator) {
				// �P�O�̊֐������X�g�ɒǉ�
				funcs.push_back(x->creator);
			}
		}
	}
}

// �֐��N���X�i���Z�j
class Add : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = xs[0];
		auto x1 = xs[1];
		auto y = (*x0) + (*x1);
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		return { gys[0], gys[0] };
	}
};

// �֐��N���X�i2��j
class Square : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = xs[0];
		auto y = nc::power(*x, 2);
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x = this->inputs[0]->data;
		auto gy = gys[0];
		auto gx = 2.0 * (*x) * (*gy);
		return { as_array(gx) };
	}
};

//----------------------------------
// function
//----------------------------------
// ���Z
VariablePtr add(const VariablePtrList& xs)
{
	return (*std::shared_ptr<Function>(new Add()))(xs)[0];
}

// 2��
VariablePtr square(const VariablePtr& xs)
{
	return (*std::shared_ptr<Function>(new Square()))(xs)[0];
}

void step14()
{
	auto x = as_variable(as_array({ 3.0 }));
	auto y = add({ x, x });
	y->backward();
	std::cout << NdArrayPrinter(x->grad) << std::endl;

	x->cleargrad();
	y = add({ add({ x, x }), x });
	y->backward();
	std::cout << NdArrayPrinter(x->grad) << std::endl;
}

}
