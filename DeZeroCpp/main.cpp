// DeZeroCpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include "dezero/steps.hpp"

using namespace dz;


namespace step01{
void step01()
{
	NdArray data = { 1.0 };
	auto x = Variable(data);
	std::cout << NdArrayPrinter(x.data) << std::endl;
}
}

namespace step02 {
void step02()
{
	auto x = Variable(NdArray({ 10.0 }));
	auto f = Square();
	auto y = f(x);
	std::cout << typeid(y).name() << std::endl;
	std::cout << NdArrayPrinter(y.data) << std::endl;
}
}

namespace step03 {
void step03()
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();

	auto x = Variable(NdArray({ 0.5 }));
	auto a = A(x);
	auto b = B(a);
	auto y = C(b);
	std::cout << NdArrayPrinter(y.data) << std::endl;
}
}

namespace step04 {
Variable f(const Variable& x)
{
	auto A = Square();
	auto B = Exp();
	auto C = Square();
	return C(B(A(x)));
}

void step04()
{
	auto x = Variable(NdArray({ 0.5 }));
	auto dy = numerical_diff(f, x);
	std::cout << NdArrayPrinter(dy) << std::endl;
}
}


int main()
{
	std::cout << std::fixed << std::setprecision(15);

	//step01::step01();
	//step02::step02();
	//step03::step03();
	step04::step04();
}
