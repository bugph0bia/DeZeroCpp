
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step02 {

//----------------------------------
// class
//----------------------------------

// 変数クラス
class Variable
{
public:
	// 内部データ
	NdArray	data;

	// コンストラクタ
	Variable(const NdArray& data) :
		data(data)
	{}

	// デストラクタ
	virtual ~Variable() {}
};

// 関数クラス
class Function
{
public:
	// デストラクタ
	virtual ~Function() {}

	// ()演算子
	Variable operator()(const Variable& input)
	{
		auto x = input.data;
		auto y = this->forward(x);
		auto output = Variable(y);
		return output;
	}

	// 順伝播
	virtual NdArray forward(const NdArray& x) = 0;
};

// 関数クラス（2乗）
class Square : public Function
{
public:
	// 順伝播
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
