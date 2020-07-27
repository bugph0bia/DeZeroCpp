#pragma once

#include "../dezero/dezero.hpp"

namespace dz
{

//----------------------------------
// class
//----------------------------------
// ŠÖ”ƒNƒ‰ƒXisinj
class Sin : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::sin(x);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto gx = gy * cos(x);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXicosj
class Cos : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::cos(x);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto gx = gy * -sin(x);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXitanhj
class Tanh : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::tanh(x);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * (1 - y * y);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXireshapej
class Reshape : public Function
{
public:
	// Œ`ó
	nc::Shape shape;
	// “ü—Íƒf[ƒ^‚ÌŒ³‚ÌŒ`ó
	nc::Shape x_shape;

	// ƒRƒ“ƒXƒgƒ‰ƒNƒ^
	Reshape(const nc::Shape& shape) :
		shape(shape)
	{}

	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = x.reshape(this->shape);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = reshape(gy, this->x_shape);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXitransposej
class Transpose : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = x.transpose();
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = transpose(gy);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXisumj
class Sum : public Function
{
public:
	// Ž²•ûŒü
	nc::Axis axis;
	// “ü—Íƒf[ƒ^‚ÌŒ³‚ÌŒ`ó
	nc::Shape x_shape;

	// ƒRƒ“ƒXƒgƒ‰ƒNƒ^
	Sum(nc::Axis axis) :
		axis(axis)
	{}

	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = x.sum(this->axis);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		//gy = reshape_sum_backward(gy, this->x_shape, this->axis);	// NdArray‚ÍŽŸŒ³”ŒÅ’è‚È‚Ì‚Å•s—v
		auto gx = broadcast_to(gy, this->x_shape);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXibroadcast_toj
class BroadcastTo : public Function
{
public:
	// Œ`ó
	nc::Shape shape;
	// “ü—Íƒf[ƒ^‚ÌŒ³‚ÌŒ`ó
	nc::Shape x_shape;

	// ƒRƒ“ƒXƒgƒ‰ƒNƒ^
	BroadcastTo(const nc::Shape& shape) :
		shape(shape)
	{}

	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = broadcast_to(x, this->shape);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = sum_to(gy, this->x_shape);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXisum_toj
class SumTo : public Function
{
public:
	// Œ`ó
	nc::Shape shape;
	// “ü—Íƒf[ƒ^‚ÌŒ³‚ÌŒ`ó
	nc::Shape x_shape;

	// ƒRƒ“ƒXƒgƒ‰ƒNƒ^
	SumTo(const nc::Shape& shape) :
		shape(shape)
	{}

	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = sum_to(x, this->shape);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = broadcast_to(gy, this->x_shape);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXimatmulj
class MatMul : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto W = *(xs[1]);
		auto y = x.dot(W);
		return { as_array(y) };
	}
	// ‹t“`”d
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
	// Œ`ó‚ª•Ï‚í‚ç‚È‚¢‚Ì‚Å‚ ‚ê‚Î‚»‚Ì‚Ü‚Ü•Ô‚·
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
	// Œ`ó‚ª•Ï‚í‚ç‚È‚¢‚Ì‚Å‚ ‚ê‚Î‚»‚Ì‚Ü‚Ü•Ô‚·
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
	// Œ`ó‚ª•Ï‚í‚ç‚È‚¢‚Ì‚Å‚ ‚ê‚Î‚»‚Ì‚Ü‚Ü•Ô‚·
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
