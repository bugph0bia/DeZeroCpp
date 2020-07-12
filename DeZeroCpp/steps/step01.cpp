
#include "pch.h"

#include "../dezero/dezero.hpp"

using namespace dz;

namespace step01 {

//----------------------------------
// class
//----------------------------------

// �ϐ��N���X
class Variable
{
public:
	// �����f�[�^
	NdArray	data;

	// �R���X�g���N�^
	Variable(const NdArray& data) :
		data(data)
	{}

	// �f�X�g���N�^
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
