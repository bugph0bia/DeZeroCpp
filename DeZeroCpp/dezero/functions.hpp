#pragma once

#include "../dezero/dezero.hpp"

namespace dz::functions
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

// ŠÖ”ƒNƒ‰ƒXiexpj
class Exp : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::exp(x);
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * y;
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
	// ²•ûŒü
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
		//gy = reshape_sum_backward(gy, this->x_shape, this->axis);	// NdArray‚ÍŸŒ³”ŒÅ’è‚È‚Ì‚Å•s—v
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
		auto y = utils::broadcast_to(x, this->shape);
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
		auto y = utils::sum_to(x, this->shape);
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

// ŠÖ”ƒNƒ‰ƒXiüŒ`•ÏŠ·/‘SŒ‹‡j
class Linear : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto W = *(xs[1]);
		auto y = x.dot(W);
		if (xs.size() >= 3 && xs[2]) {
			auto b = *(xs[2]);
			utils::broadcast_mutual(y, b);	// NdArray‚Ìl‘¥‰‰Z‘O‚Ìƒuƒ[ƒhƒLƒƒƒXƒg
			y = y + b;
		}
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto W = this->inputs[1];
		auto b = this->inputs[2];
		auto gy = gys[0];
		auto gb = as_variable(nullptr);
		if (b->data) {
			gb = sum_to(gy, b->shape());
		}
		auto gx = matmul(gy, W->transpose());
		auto gW = matmul(x->transpose(), gy);
		return { gx, gW, gb };
	}
};

// ŠÖ”ƒNƒ‰ƒXiƒVƒOƒ‚ƒCƒhj
class Sigmoid : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		//auto y = 1.0 / (1.0 + nc::exp(x));
		auto y = nc::tanh(x * 0.5) * 0.5 + 0.5;	// ‚æ‚è—Ç‚¢À‘••û–@
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * y * (1.0 - y);
		return { gx };
	}
};

// ŠÖ”ƒNƒ‰ƒXi•½‹Ï“ñæŒë·j
class MeanSquaredError : public Function
{
public:
	// ‡“`”d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto diff = x0 - x1;
		auto y = nc::power(diff, 2).sum() / static_cast<data_t>(diff.size());
		return { as_array(y) };
	}
	// ‹t“`”d
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x0 = this->inputs[0];
		auto x1 = this->inputs[1];
		auto gy = gys[0];
		auto diff = x0 - x1;
		gy = broadcast_to(gy, diff->shape());
		auto gx0 = gy * diff * (2.0 / diff->size());
		auto gx1 = -gx0;
		return { gx0, gx1 };
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
inline VariablePtrList sin(const VariablePtrList& xs)
{
	return { sin(xs[0]) };
}

// cos
inline VariablePtr cos(const VariablePtr& x)
{
	auto f = FunctionPtr(new Cos());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList cos(const VariablePtrList& xs)
{
	return { cos(xs[0]) };
}

// tanh
inline VariablePtr tanh(const VariablePtr& x)
{
	auto f = FunctionPtr(new Tanh());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList tanh(const VariablePtrList& xs)
{
	return { tanh(xs[0]) };
}

// exp
inline VariablePtr exp(const VariablePtr& x)
{
	auto f = FunctionPtr(new Exp());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList exp(const VariablePtrList& xs)
{
	return { exp(xs[0]) };
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
inline VariablePtrList reshape(const VariablePtrList& xs, const nc::Shape& shape)
{
	return { reshape(xs[0], shape) };
}

// transpose
inline VariablePtr transpose(const VariablePtr& x)
{
	auto f = FunctionPtr(new Transpose());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList transpose(const VariablePtrList& xs)
{
	return { transpose(xs[0]) };
}

// sum
inline VariablePtr sum(const VariablePtr& x, nc::Axis axis /*=nc::Axis::NONE*/)
{
	auto f = FunctionPtr(new Sum(axis));
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList sum(const VariablePtrList& xs, nc::Axis axis /*=nc::Axis::NONE*/)
{
	return { sum(xs[0], axis) };
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
inline VariablePtrList broadcast_to(const VariablePtrList& xs, const nc::Shape& shape)
{
	return { broadcast_to(xs[0], shape) };
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
inline VariablePtrList sum_to(const VariablePtrList& xs, const nc::Shape& shape)
{
	return { sum_to(xs[0], shape) };
}

// matmul
inline VariablePtr matmul(const VariablePtr& x, const VariablePtr& W)
{
	auto f = FunctionPtr(new MatMul());
	VariablePtrList args = { x, W };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList matmul(const VariablePtrList& xs)
{
	return { matmul(xs[0], xs[1]) };
}

// linear
inline VariablePtr linear(const VariablePtr& x, const VariablePtr& W, const VariablePtr& b /*=nullptr*/)
{
	auto f = FunctionPtr(new Linear());
	VariablePtrList args = { x, W, b };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList linear(const VariablePtrList& xs)
{
	if (xs.size() >= 3)
		return { linear(xs[0], xs[1], xs[2]) };
	else
		return { linear(xs[0], xs[1]) };
}

// linear ŠÈˆÕ”Å
inline VariablePtr linear_simple(const VariablePtr& x, const VariablePtr& W, const VariablePtr& b /*=nullptr*/)
{
	auto t = matmul(x, W);
	if (!b) return t;

	auto y = t + b;
	t->data = nullptr;	// t‚Ìƒf[ƒ^‚Í•s—v‚È‚Ì‚ÅÁ‹
	return y;
}
inline VariablePtrList linear_simple(const VariablePtrList& xs)
{
	if (xs.size() >= 3)
		return { linear_simple(xs[0], xs[1], xs[2]) };
	else
		return { linear_simple(xs[0], xs[1]) };
}

// sigmoid
inline VariablePtr sigmoid(const VariablePtr& x)
{
	auto f = FunctionPtr(new Sigmoid());
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList sigmoid(const VariablePtrList& xs)
{
	return { sigmoid(xs[0]) };
}

// sigmoid ŠÈˆÕ”Å
inline VariablePtr sigmoid_simple(const VariablePtr& x)
{
	auto y = 1.0 / (1.0 + exp(-x));
	return y;
}
inline VariablePtrList sigmoid_simple(const VariablePtrList& xs)
{
	return { sigmoid_simple(xs[0]) };
}

// mean_squared_error
inline VariablePtr mean_squared_error(const VariablePtr& x0, const VariablePtr& x1)
{
	auto f = FunctionPtr(new MeanSquaredError());
	VariablePtrList args = { x0, x1 };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList mean_squared_error(const VariablePtrList& xs)
{
	return { mean_squared_error(xs[0], xs[1]) };
}

}	// namespace dz::functions
