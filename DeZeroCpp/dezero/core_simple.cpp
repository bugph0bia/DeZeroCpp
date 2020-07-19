
#include "pch.h"

#include "core_simple.hpp"

namespace dz
{

//----------------------------------
// type
//----------------------------------

// NdArrayPtr�����֐�
NdArrayPtr as_array(nullptr_t /*=nullptr*/)
{
	return NdArrayPtr();	// �����Ȃ��܂��� nullptr �̏ꍇ�� Empty �Ƃ���
}
NdArrayPtr as_array(std::initializer_list<NdArray::value_type> list)
{
	return std::make_shared<NdArray>(list);
}
NdArrayPtr as_array(NdArray::value_type scalar)
{
	return as_array({ scalar });
}
NdArrayPtr as_array(const NdArray& data)
{
	return std::make_shared<NdArray>(data);
}

// VariablePtr�����֐�
VariablePtr as_variable(nullptr_t /*=nullptr*/)
{
	return VariablePtr();	// �����Ȃ��܂��� nullptr �̏ꍇ�� Empty �Ƃ���
}
VariablePtr as_variable(const NdArrayPtr& data)
{
	return std::make_shared<Variable>(data);
}
VariablePtr as_variable(const Variable& data)
{
	return std::make_shared<Variable>(data);
}

//----------------------------------
// class
//----------------------------------

std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda)
{
	// nullptr �̏ꍇ
	if (!nda.data) ost << "Null";
	// NdArray���X�J���[�Ȃ璆�g�̃f�[�^��W���o�͂�
	else if (nda.data->shape().rows == 1 && nda.data->shape().cols == 1) ost << (*nda.data)[0];
	// �ʏ펞
	else ost << nda.data;
	return ost;
}

std::ostream& operator<<(std::ostream& ost, const Variable& v)
{
	std::ostringstream osst;
	// �W���o�͂̏����_�ȉ������� 15 �Ƃ���
	osst << std::fixed << std::setprecision(15);
	osst << NdArrayPrinter(v.data);
	auto str = osst.str();

	// �����̉��s���폜
	if (str.back() == '\n') str.pop_back();

	// �r���̉��s�ɃC���f���g��ǉ�
	str = replace_all(str, "\n", "\n          ");

	ost << "variable(" << str << ")";
	return ost;
}

std::ostream& operator<<(std::ostream& ost, const VariablePtr& p)
{
	if (!p) ost << "variable(Null)";
	else ost << *p;
	return ost;
}

// �������̊֐���ݒ�
void Variable::set_creator(const FunctionPtr& func)
{
	creator = func;

	// �������̊֐��̐���� +1 ���Ď��g�̐���Ƃ���
	this->generation = func->generation + 1;
}

// �t�`�d
// ������ Function �N���X�̃����o���Q�Ƃ��Ă��邽�߂��̈ʒu�Œ�`����K�v������
void Variable::backward(bool retain_grad /*=false*/)
{
	// ���z�����ݒ聁�t�`�d�̊J�n�_
	if (!this->grad) {
		// ���z�̏����l(1)��ݒ�
		auto g = nc::ones_like<data_t>(*this->data);
		this->grad = as_array(g);
	}

	// �֐����X�g
	auto funcs = std::list<FunctionPtr>();
	// �����ς݊֐��Z�b�g
	auto seen_set = std::set<FunctionPtr>();

	// �N���[�W���F�֐����X�g�֒ǉ�
	auto add_func = [&funcs, &seen_set](const FunctionPtr& f) {
		// ���X�g�֖��ǉ��̊֐��Ȃ�
		if (seen_set.find(f) == seen_set.end()) {
			// ���X�g�֒ǉ����Đ���ŏ����\�[�g����
			funcs.push_back(f);
			seen_set.insert(f);
			funcs.sort([](const FunctionPtr& lhs, const FunctionPtr& rhs) { return lhs->generation < rhs->generation; });
		}
	};

	// �ŏ��̊֐������X�g�ɒǉ�
	add_func(this->creator);

	// �֐����X�g����ɂȂ�܂Ń��[�v
	while (!funcs.empty()) {
		// ���X�g����֐������o��
		auto f = funcs.back();
		funcs.pop_back();

		// �o�̓f�[�^������z�����o��
		auto gys = NdArrayPtrList();
		for (const auto& o : f->outputs) {
			gys.push_back(o.lock()->grad);
		}

		// �t�`�d
		auto gxs = f->backward(gys);

		// ���̓f�[�^�ƎZ�o�������z�̗v�f���͈�v����K�v����
		assert(f->inputs.size() == gxs.size());

		// ���̓f�[�^�ƌ��z�̃y�A�����[�v
		for (size_t i = 0; i < gxs.size(); i++) {
			auto x = f->inputs[i];
			auto gx = gxs[i];

			// ���z�����ݒ�Ȃ�������
			if (!x->grad) {
				x->grad = gx;
			}
			// ���z���ݒ�ς݂Ȃ���Z����
			else {
				// �V���� NdArrayPtr �C���X�^���X����邱�Ƃ��d�v
				// �Ⴆ�΁A*x->grad += *gx; �Ƃ��Ă͂����Ȃ��i�t�^A�Q�Ɓj
				x->grad = as_array(*x->grad + *gx);
			}

			// �P�O�̊֐������X�g�ɒǉ�
			if (x->creator) {
				add_func(x->creator);
			}
		}

		// ���z��ێ����Ȃ��ꍇ
		if (!retain_grad) {
			// ���z���폜
			for (const auto& y : f->outputs) {
				y.lock()->grad = nullptr;
			}
		}
	}
}

//----------------------------------
// function
//----------------------------------

// ���Z
VariablePtr add(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Add()))({ x0, x1 })[0];
}

// ���Z
VariablePtr sub(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Sub()))({ x0, x1 })[0];
}

// ��Z
VariablePtr mul(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Mul()))({ x0, x1 })[0];
}

// ���Z
VariablePtr div(const VariablePtr& x0, const VariablePtr& x1)
{
	return (*std::shared_ptr<Function>(new Div()))({ x0, x1 })[0];
}

// ����
VariablePtr pos(const VariablePtr& x)
{
	return (*std::shared_ptr<Function>(new Pos()))({ x })[0];
}

// ����
VariablePtr neg(const VariablePtr& x)
{
	return (*std::shared_ptr<Function>(new Neg()))({ x })[0];
}

// �ݏ�
VariablePtr power(const VariablePtr& x0, uint32_t c)
{
	return (*std::shared_ptr<Function>(new Pow(c)))(x0)[0];
}
VariablePtr power(const NdArrayPtr& x, uint32_t c)
{
	return power(as_variable(x), c);
}
VariablePtr power(data_t x, uint32_t c)
{
	return power(as_variable(as_array(x)), c);
}

// 2��
VariablePtr square(const VariablePtr& x0)
{
	return (*std::shared_ptr<Function>(new Square()))(x0)[0];
}

// VariablePtr�̉��Z�q�I�[�o�[���[�h
// �񍀉��Z�q +
VariablePtr operator+(const VariablePtr& lhs, const VariablePtr& rhs) { return add(lhs, rhs); }
VariablePtr operator+(const VariablePtr& lhs, const NdArrayPtr& rhs) { return add(lhs, as_variable(rhs)); }
VariablePtr operator+(const NdArrayPtr& lhs, const VariablePtr& rhs) { return add(as_variable(lhs), rhs); }
VariablePtr operator+(const VariablePtr& lhs, data_t rhs) { return add(lhs, as_variable(as_array(rhs))); }
VariablePtr operator+(data_t lhs, const VariablePtr& rhs) { return add(as_variable(as_array(lhs)), rhs); }
// �񍀉��Z�q -
VariablePtr operator-(const VariablePtr& lhs, const VariablePtr& rhs) { return sub(lhs, rhs); }
VariablePtr operator-(const VariablePtr& lhs, const NdArrayPtr& rhs) { return sub(lhs, as_variable(rhs)); }
VariablePtr operator-(const NdArrayPtr& lhs, const VariablePtr& rhs) { return sub(as_variable(lhs), rhs); }
VariablePtr operator-(const VariablePtr& lhs, data_t rhs) { return sub(lhs, as_variable(as_array(rhs))); }
VariablePtr operator-(data_t lhs, const VariablePtr& rhs) { return sub(as_variable(as_array(lhs)), rhs); }
// �񍀉��Z�q *
VariablePtr operator*(const VariablePtr& lhs, const VariablePtr& rhs) { return mul(lhs, rhs); }
VariablePtr operator*(const VariablePtr& lhs, const NdArrayPtr& rhs) { return mul(lhs, as_variable(rhs)); }
VariablePtr operator*(const NdArrayPtr& lhs, const VariablePtr& rhs) { return mul(as_variable(lhs), rhs); }
VariablePtr operator*(const VariablePtr& lhs, data_t rhs) { return mul(lhs, as_variable(as_array(rhs))); }
VariablePtr operator*(data_t lhs, const VariablePtr& rhs) { return mul(as_variable(as_array(lhs)), rhs); }
// �񍀉��Z�q /
VariablePtr operator/(const VariablePtr& lhs, const VariablePtr& rhs) { return div(lhs, rhs); }
VariablePtr operator/(const VariablePtr& lhs, const NdArrayPtr& rhs) { return div(lhs, as_variable(rhs)); }
VariablePtr operator/(const NdArrayPtr& lhs, const VariablePtr& rhs) { return div(as_variable(lhs), rhs); }
VariablePtr operator/(const VariablePtr& lhs, data_t rhs) { return div(lhs, as_variable(as_array(rhs))); }
VariablePtr operator/(data_t lhs, const VariablePtr& rhs) { return div(as_variable(as_array(lhs)), rhs); }
// �P�����Z�q +
VariablePtr operator+(const VariablePtr& data) { return pos(data); }
// �P�����Z�q -
VariablePtr operator-(const VariablePtr& data) { return neg(data); }

}	// namespace dezerocpp
