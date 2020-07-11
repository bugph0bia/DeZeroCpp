#pragma once

#include <iostream>
#include <cmath>
#include <functional>

#include "NumCpp.hpp"


namespace dz
{

//----------------------------------
// typedef
//----------------------------------
using data_t = double;	// TODO: �ŏI�I�ɂ� float �ɂ���
using NdArray = nc::NdArray<data_t>;

//----------------------------------
// utility
//----------------------------------
// NdArray�̏o�̓w���p�[�N���X
class NdArrayPrinter
{
public:
	NdArray& data;
	NdArrayPrinter(NdArray& data) :
		data(data)
	{}
};
std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// NdArray���X�J���[�Ȃ璆�g�̃f�[�^��W���o�͂�
	if (nda.data.shape().rows == 1 && nda.data.shape().cols == 1) ost << nda.data[0];
	else ost << nda.data;
	return ost;
}

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

// �֐��N���X�iexp�j
class Exp : public Function
{
public:
	// ���`�d
	NdArray forward(const NdArray& x) override
	{
		return nc::exp(x);
	}
};

//----------------------------------
// function
//----------------------------------
// ���l����
NdArray numerical_diff(std::function<Variable(Variable)> f, const Variable& x, data_t eps = 1e-4)
{
	auto x0 = Variable(x.data - eps);
	auto x1 = Variable(x.data + eps);
	auto y0 = f(x0);
	auto y1 = f(x1);
	return (y1.data - y0.data) / (2 * eps);
}

}	// namespace dezerocpp
