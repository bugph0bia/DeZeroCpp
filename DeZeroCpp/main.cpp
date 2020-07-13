// DeZeroCpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"

#include "dezero/dezero.hpp"

#include <iomanip>

namespace step01 { extern void step01(); }
namespace step02 { extern void step02(); }
namespace step03 { extern void step03(); }
namespace step04 { extern void step04(); }
//namespace step05 { extern void step05(); }
namespace step06 { extern void step06(); }
namespace step07 { extern void step07(); }
namespace step08 { extern void step08(); }

int main()
{
	// 標準出力の小数点以下桁数を 15 とする
	std::cout << std::fixed << std::setprecision(15);

	step08::step08();
}
