
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step06 {

//----------------------------------
// class
//----------------------------------

// �ϐ��N���X
class Variable
{
public:
	// �����f�[�^
	NdArray	data;
	// ���z
	std::shared_ptr<NdArray> grad;

	// �R���X�g���N�^
	Variable(const NdArray& data) :
		data(data)
	{}

	// �f�X�g���N�^
	virtual ~Variable() {}
};

// �֐��N���X
class Function
{
protected:
	// ���̓f�[�^
	std::shared_ptr<Variable> input;

public:
	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	Variable operator()(const Variable& input)
	{
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		this->input = std::make_shared<Variable>(input);
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

//----------------------------------
// function
//----------------------------------

void step06()
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);

	y.grad = std::make_shared<NdArray>(NdArray({ 1.0 }));
	b.grad = std::make_shared<NdArray>(C.backward(*y.grad));
	a.grad = std::make_shared<NdArray>(B.backward(*b.grad));
	x.grad = std::make_shared<NdArray>(A.backward(*a.grad));

	std::cout << NdArrayPrinter(*x.grad);
}

}
