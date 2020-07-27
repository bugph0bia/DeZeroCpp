#pragma once

#include "../dezero/dezero.hpp"

namespace dz
{

//----------------------------------
// class
//----------------------------------
// �֐��N���X�isin�j
class Sin : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::sin(x);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto gx = gy * cos(x);
		return { gx };
	}
};

// �֐��N���X�icos�j
class Cos : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::cos(x);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto gx = gy * -sin(x);
		return { gx };
	}
};

// �֐��N���X�itanh�j
class Tanh : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::tanh(x);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * (1 - y * y);
		return { gx };
	}
};

// �֐��N���X�ireshape�j
class Reshape : public Function
{
public:
	// �`��
	nc::Shape shape;
	// ���̓f�[�^�̌��̌`��
	nc::Shape x_shape;

	// �R���X�g���N�^
	Reshape(const nc::Shape& shape) :
		shape(shape)
	{}

	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = x.reshape(this->shape);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = reshape(gy, this->x_shape);
		return { gx };
	}
};

// �֐��N���X�itranspose�j
class Transpose : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = x.transpose();
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = transpose(gy);
		return { gx };
	}
};

// �֐��N���X�isum�j
class Sum : public Function
{
public:
	// ������
	nc::Axis axis;
	// ���̓f�[�^�̌��̌`��
	nc::Shape x_shape;

	// �R���X�g���N�^
	Sum(nc::Axis axis) :
		axis(axis)
	{}

	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = x.sum(this->axis);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		//gy = reshape_sum_backward(gy, this->x_shape, this->axis);	// NdArray�͎������Œ�Ȃ̂ŕs�v
		auto gx = broadcast_to(gy, this->x_shape);
		return { gx };
	}
};

// �֐��N���X�ibroadcast_to�j
class BroadcastTo : public Function
{
public:
	// �`��
	nc::Shape shape;
	// ���̓f�[�^�̌��̌`��
	nc::Shape x_shape;

	// �R���X�g���N�^
	BroadcastTo(const nc::Shape& shape) :
		shape(shape)
	{}

	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = broadcast_to(x, this->shape);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = sum_to(gy, this->x_shape);
		return { gx };
	}
};

// �֐��N���X�isum_to�j
class SumTo : public Function
{
public:
	// �`��
	nc::Shape shape;
	// ���̓f�[�^�̌��̌`��
	nc::Shape x_shape;

	// �R���X�g���N�^
	SumTo(const nc::Shape& shape) :
		shape(shape)
	{}

	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = sum_to(x, this->shape);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = broadcast_to(gy, this->x_shape);
		return { gx };
	}
};

// �֐��N���X�imatmul�j
class MatMul : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto W = *(xs[1]);
		auto y = x.dot(W);
		return { as_array(y) };
	}
	// �t�`�d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto W = this->inputs[1];
		auto gy = gys[0];
		auto gx = matmul(gy, W->transpose());
		auto gW = matmul(x->transpose(), gy);
		return { gx, gW };
	}
};

//----------------------------------
// function
//----------------------------------

// sin
inline VariablePtr sin(const VariablePtr& x)
{
	auto f = FunctionPtr(new Sin());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// cos
inline VariablePtr cos(const VariablePtr& x)
{
	auto f = FunctionPtr(new Cos());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// tanh
inline VariablePtr tanh(const VariablePtr& x)
{
	auto f = FunctionPtr(new Tanh());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// reshape
inline VariablePtr reshape(const VariablePtr& x, const nc::Shape& shape)
{
	// �`�󂪕ς��Ȃ��̂ł���΂��̂܂ܕԂ�
	if (x->data->shape() == shape) {
		return as_variable(*x);
	}
	auto f = FunctionPtr(new Reshape(shape));
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// transpose
inline VariablePtr transpose(const VariablePtr& x)
{
	auto f = FunctionPtr(new Transpose());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// sum
inline VariablePtr sum(const VariablePtr& x, nc::Axis axis /*=nc::Axis::NONE*/)
{
	auto f = FunctionPtr(new Sum(axis));
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// bloadcast_to
inline VariablePtr broadcast_to(const VariablePtr& x, const nc::Shape& shape)
{
	// �`�󂪕ς��Ȃ��̂ł���΂��̂܂ܕԂ�
	if (x->data->shape() == shape) {
		return as_variable(*x);
	}
	auto f = FunctionPtr(new BroadcastTo(shape));
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// bloadcast_to
inline VariablePtr sum_to(const VariablePtr& x, const nc::Shape& shape)
{
	// �`�󂪕ς��Ȃ��̂ł���΂��̂܂ܕԂ�
	if (x->data->shape() == shape) {
		return as_variable(*x);
	}
	auto f = FunctionPtr(new SumTo(shape));
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}

// matmul
inline VariablePtr matmul(const VariablePtr& x, const VariablePtr& W)
{
	auto f = FunctionPtr(new MatMul());
	VariablePtrList args = { x, W };
	auto ys = (*f)(args);
	return ys[0];
}

}	// namespace dz
