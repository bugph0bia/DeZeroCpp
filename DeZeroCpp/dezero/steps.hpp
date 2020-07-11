#pragma once

#include <cmath>
#include "NumCpp.hpp"

namespace dz
{

//----------------------------------
// typedef
//----------------------------------
using data_t = float;
using NdArray = nc::NdArray<data_t>;

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
	// ()���Z�q
	Variable operator()(const Variable& input) {
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
	NdArray forward(const NdArray& x)
	{
		return nc::power(x, 2);
	}
};

}	// namespace dezerocpp
