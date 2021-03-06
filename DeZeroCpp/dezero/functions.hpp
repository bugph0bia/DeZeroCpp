#pragma once

#include "../dezero/dezero.hpp"

namespace dz::functions
{

//----------------------------------
// type
//----------------------------------

// 汎用的な関数型
using function_t = VariablePtrList(const VariablePtrList&);

//----------------------------------
// class
//----------------------------------

// 関数クラス（sin）
class Sin : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::sin(x);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto gx = gy * cos(x);
		return { gx };
	}
};

// 関数クラス（cos）
class Cos : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::cos(x);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto x = this->inputs[0];
		auto gy = gys[0];
		auto gx = gy * -sin(x);
		return { gx };
	}
};

// 関数クラス（tanh）
class Tanh : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::tanh(x);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * (1 - y * y);
		return { gx };
	}
};

// 関数クラス（exp）
class Exp : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::exp(x);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * y;
		return { gx };
	}
};

// 関数クラス（reshape）
class Reshape : public Function
{
public:
	// 形状
	nc::Shape shape;
	// 入力データの元の形状
	nc::Shape x_shape;

	// コンストラクタ
	Reshape(const nc::Shape& shape) :
		shape(shape)
	{}

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = x.reshape(this->shape);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = reshape(gy, this->x_shape);
		return { gx };
	}
};

// 関数クラス（transpose）
class Transpose : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = x.transpose();
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = transpose(gy);
		return { gx };
	}
};

// 関数クラス（sum）
class Sum : public Function
{
public:
	// 軸方向
	nc::Axis axis;
	// 入力データの元の形状
	nc::Shape x_shape;

	// コンストラクタ
	Sum(nc::Axis axis) :
		axis(axis)
	{}

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = x.sum(this->axis);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		//gy = reshape_sum_backward(gy, this->x_shape, this->axis);	// NdArrayは次元数固定なので不要
		auto gx = broadcast_to(gy, this->x_shape);
		return { gx };
	}
};

// 関数クラス（broadcast_to）
class BroadcastTo : public Function
{
public:
	// 形状
	nc::Shape shape;
	// 入力データの元の形状
	nc::Shape x_shape;

	// コンストラクタ
	BroadcastTo(const nc::Shape& shape) :
		shape(shape)
	{}

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = utils::broadcast_to(x, this->shape);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = sum_to(gy, this->x_shape);
		return { gx };
	}
};

// 関数クラス（sum_to）
class SumTo : public Function
{
public:
	// 形状
	nc::Shape shape;
	// 入力データの元の形状
	nc::Shape x_shape;

	// コンストラクタ
	SumTo(const nc::Shape& shape) :
		shape(shape)
	{}

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		this->x_shape = x.shape();
		auto y = utils::sum_to(x, this->shape);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto gx = broadcast_to(gy, this->x_shape);
		return { gx };
	}
};

// 関数クラス（matmul）
class MatMul : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto W = *(xs[1]);
		auto y = x.dot(W);
		return { as_array(y) };
	}
	// 逆伝播
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

// 関数クラス（線形変換/全結合）
class Linear : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto W = *(xs[1]);
		auto y = x.dot(W);
		if (xs.size() >= 3 && xs[2]) {
			auto b = *(xs[2]);
			utils::broadcast_mutual(y, b);	// NdArrayの四則演算前のブロードキャスト
			y = y + b;
		}
		return { as_array(y) };
	}
	// 逆伝播
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

// 関数クラス（シグモイド）
class Sigmoid : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		//auto y = 1.0 / (1.0 + nc::exp(x));
		auto y = nc::tanh(x * 0.5) * 0.5 + 0.5;	// より良い実装方法
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto y = this->outputs[0].lock();
		auto gy = gys[0];
		auto gx = gy * y * (1.0 - y);
		return { gx };
	}
};

// 関数クラス（平均二乗誤差）
class MeanSquaredError : public Function
{
public:
	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto diff = x0 - x1;
		auto y = nc::power(diff, 2).sum() / static_cast<data_t>(diff.size());
		return { as_array(y) };
	}
	// 逆伝播
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

// 関数クラス（Softmax）
class Softmax : public Function
{
public:
	// 軸
	nc::Axis axis;

	// コンストラクタ
	Softmax(nc::Axis axis = nc::Axis::ROW) :
		axis(axis)
	{}

	// 順伝播
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = x - x.max(axis);
		y = nc::exp(y);
		y /= y.sum(axis);
		return { as_array(y) };
	}
	// 逆伝播
	VariablePtrList backward(const VariablePtrList& gys) override
	{
		auto gy = gys[0];
		auto y = this->outputs[0].lock();
		auto gx = y * gy;
		auto sumdx = gx->sum(axis);
		gx = gx - y * sumdx;
		return { gx };
	}
};

//----------------------------------
// function
//----------------------------------

// sin
inline VariablePtr sin(const VariablePtr& x)
{
	FunctionPtr f = std::make_shared<Sin>();
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
	FunctionPtr f = std::make_shared<Cos>();
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
	FunctionPtr f = std::make_shared<Tanh>();
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
	FunctionPtr f = std::make_shared<Exp>();
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
	// 形状が変わらないのであればそのまま返す
	if (x->data->shape() == shape) {
		return as_variable(*x);
	}
	FunctionPtr f = std::make_shared<Reshape>(shape);
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
	FunctionPtr f = std::make_shared<Transpose>();
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
	FunctionPtr f = std::make_shared<Sum>(axis);
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
	// 形状が変わらないのであればそのまま返す
	if (x->data->shape() == shape) {
		return as_variable(*x);
	}
	FunctionPtr f = std::make_shared<BroadcastTo>(shape);
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
	// 形状が変わらないのであればそのまま返す
	if (x->data->shape() == shape) {
		return as_variable(*x);
	}
	FunctionPtr f = std::make_shared<SumTo>(shape);
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
	FunctionPtr f = std::make_shared<MatMul>();
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
	FunctionPtr f = std::make_shared<Linear>();
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

// linear 簡易版
inline VariablePtr linear_simple(const VariablePtr& x, const VariablePtr& W, const VariablePtr& b /*=nullptr*/)
{
	auto t = matmul(x, W);
	if (!b) return t;

	auto y = t + b;
	t->data = nullptr;	// tのデータは不要なので消去
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
	FunctionPtr f = std::make_shared<Sigmoid>();
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList sigmoid(const VariablePtrList& xs)
{
	return { sigmoid(xs[0]) };
}

// sigmoid 簡易版
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
	FunctionPtr f = std::make_shared<MeanSquaredError>();
	VariablePtrList args = { x0, x1 };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList mean_squared_error(const VariablePtrList& xs)
{
	return { mean_squared_error(xs[0], xs[1]) };
}

// softmax
inline VariablePtr softmax(const VariablePtr& x, nc::Axis axis /*=nc::Axis::ROW*/)
{
	FunctionPtr f = std::make_shared<Softmax>(axis);
	VariablePtrList args = { x };
	auto ys = (*f)(args);
	return ys[0];
}
inline VariablePtrList softmax(const VariablePtrList& xs, nc::Axis axis)
{
	return { softmax(xs[0], axis) };
}

// softmax簡易版
inline VariablePtr softmax_simple(const VariablePtr& x, nc::Axis axis /*=nc::Axis::ROW*/)
{
	auto y = exp(x);
	auto sum_y = sum(y, axis);
	return y / sum_y;
}
inline VariablePtrList softmax_simple(const VariablePtrList& xs, nc::Axis axis)
{
	return { softmax_simple(xs[0], axis) };
}

}	// namespace dz::functions
