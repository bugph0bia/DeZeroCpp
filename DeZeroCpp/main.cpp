// DeZeroCpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include "dezero/steps.hpp"

void print_separator(const std::string& title)
{
	std::cout << std::endl << "---------------- " << title << " ----------------" << std::endl;
}

void step01()
{
	print_separator(__func__);

	dz::NdArray data = { 1.0 };
	auto x = dz::Variable(data);
	std::cout << x.data;
}

void step02()
{
	print_separator(__func__);

	auto x = dz::Variable(dz::NdArray({ 10.0 }));
	auto f = dz::Square();
	auto y = f(x);
	std::cout << typeid(y).name() << std::endl;
	std::cout << y.data << std::endl;
}

int main()
{
	step01();
	step02();
}
