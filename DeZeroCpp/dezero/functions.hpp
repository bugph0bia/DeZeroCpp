#pragma once

#include "../dezero/dezero.hpp"

namespace dz
{

extern inline VariablePtr sin(const VariablePtr& x);
extern inline VariablePtr cos(const VariablePtr& x);

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



//----------------------------------
// function
//----------------------------------



}	// namespace dz
