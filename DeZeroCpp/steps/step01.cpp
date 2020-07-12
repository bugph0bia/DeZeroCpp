
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step01 {

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

//----------------------------------
// function
//----------------------------------

void step01()
{
	NdArray data = { 1.0 };
	auto x = Variable(data);
	std::cout << NdArrayPrinter(x.data) << std::endl;
}

}
