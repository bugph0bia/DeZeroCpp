#pragma once

#include "NumCpp.hpp"

namespace dz
{

//----------------------------------
// typedef
//----------------------------------
using data_t = float;
using NdArray = nc::NdArray<data_t>;

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
	{
	}

	// �f�X�g���N�^
	virtual ~Variable() {}
};

}	// namespace dezerocpp
