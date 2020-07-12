
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step02 {

//----------------------------------
// class
//----------------------------------

// �ϐ��N���X
class Variable
{
public:
	// �����f�[�^
	NdArray	data;

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
public:
	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	Variable operator()(const Variable& input)
	{
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		return output;
	}

	// ���`�d
	virtual NdArray forward(const NdArray& x) = 0;
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
};

//----------------------------------
// function
//----------------------------------

void step02()
{
	auto x = Variable(NdArray({ 10.0 }));
	auto f = Square();
	auto y = f(x);
	std::cout << typeid(y).name() << std::endl;
	std::cout << NdArrayPrinter(y.data) << std::endl;
}

}
