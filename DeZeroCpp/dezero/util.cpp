#pragma once

#include "pch.h"

#include <iostream>

namespace dz
{

std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// NdArray���X�J���[�Ȃ璆�g�̃f�[�^��W���o�͂�
	if (nda.data.shape().rows == 1 && nda.data.shape().cols == 1) ost << nda.data[0];
	else ost << nda.data;
	return ost;
}

}
