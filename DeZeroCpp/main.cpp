// DeZeroCpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include "core.hpp"

int main()
{
    std::cout << "Hello World!\n";

	dz::NdArray a = { {1, 2, 3}, {4, 5, 6} };
	dz::Variable va(a);
	std::cout << "Variable: " << va.data << "shape: " << va.data.shape();
}
