// DeZeroCpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"

#include "dezero/dezero.hpp"

#include <iomanip>

namespace step04 {
extern void step04();
}

int main()
{
	// 標準出力の小数点以下桁数を 15 とする
	std::cout << std::fixed << std::setprecision(15);

	step04::step04();
}
