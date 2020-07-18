#pragma once

#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <list>
#include <vector>
#include <set>
#include <map>
#include "NumCpp.hpp"

namespace dz
{

// �N���X�O���錾
class Variable;
class Function;

//----------------------------------
// type
//----------------------------------

// ��{�f�[�^�^
using data_t = double;	// TODO: �ŏI�I�ɂ� float �ɂ���
using NdArray = nc::NdArray<data_t>;

// �X�}�[�g�|�C���^�^
using NdArrayPtr = std::shared_ptr<NdArray>;	// �C���X�^���X�������� std::make_shared<NdArray> �֐����g������
using VariablePtr = std::shared_ptr<Variable>;	// �C���X�^���X�������� std::make_shared<Variable> �֐����g������
using VariableWPtr = std::weak_ptr<Variable>;
using FunctionPtr = std::shared_ptr<Function>;	// �h���N���X�̃C���X�^���X�������� new ���g������
												// �imake_shared ���g���� Function �N���X���C���X�^���X������ăG���[�ƂȂ�j
// ���X�g�^
using NdArrayPtrList = std::vector<NdArrayPtr>;
using VariablePtrList = std::vector<VariablePtr>;
using VariableWPtrList = std::vector<VariableWPtr>;

// NdArrayPtr�����֐�
extern NdArrayPtr as_array(nullptr_t = nullptr);
extern NdArrayPtr as_array(std::initializer_list<NdArray::value_type> list);
extern NdArrayPtr as_array(NdArray::value_type scalar);
extern NdArrayPtr as_array(const NdArray& data);

// VariablePtr�����֐�
extern VariablePtr as_variable(nullptr_t = nullptr);
extern VariablePtr as_variable(const NdArrayPtr& data);
extern VariablePtr as_variable(const Variable& data);

//----------------------------------
// class
//----------------------------------

// �ݒ�N���X
class Config
{
private:
	// �R���X�g���N�^
	Config() {
		// �t�`�d��
		param["enable_backprop"] = true;
	}

public:
	// �ݒ�l
	std::map<std::string, bool> param;

	// �R�s�[/���[�u�s��
	Config(const Config&) = delete;
	Config(Config&&) = delete;
	Config& operator=(const Config&) = delete;
	Config& operator=(Config&&) = delete;

	// �C���X�^���X�擾
	static Config& get_instance() {
		static Config instance;
		return instance;
	}
};

// �ݒ�ꎞ�ύX�N���X
class UsingConfig
{
private:
	// �ύX�O�̒l
	std::string name;
	bool old_value;

public:
	// �R���X�g���N�^
	UsingConfig(std::string name, bool value) :
		name(name)
	{
		// �ݒ�ύX
		old_value = Config::get_instance().param[name];
		Config::get_instance().param[name] = value;
	}
	// �f�X�g���N�^
	virtual ~UsingConfig()
	{
		// �ݒ蕜��
		Config::get_instance().param[name] = old_value;
	}

	// �R�s�[/���[�u�s��
	UsingConfig(const UsingConfig&) = delete;
	UsingConfig(UsingConfig&&) = delete;
	UsingConfig& operator=(const UsingConfig&) = delete;
	UsingConfig& operator=(UsingConfig&&) = delete;
};

// �t�`�d�ۂ��ꎞ�I��OFF
struct no_grad : UsingConfig
{
	no_grad() : UsingConfig("enable_backprop", false) {}
};

// NdArray�̏o�̓w���p�[�N���X
class NdArrayPrinter
{
public:
	const std::shared_ptr<NdArray> data;
	NdArrayPrinter(const std::shared_ptr<NdArray>& data) :
		data(data)
	{}
	NdArrayPrinter(const NdArray& data) :
		data(std::make_shared<NdArray>(data))
	{}
};

extern std::ostream& operator<<(std::ostream& ost, const NdArrayPrinter& nda);

// �ϐ��N���X
class Variable
{
public:
	// �����f�[�^
	NdArrayPtr data;
	// ����
	std::string name;
	// ���z
	NdArrayPtr grad;
	// �������̊֐�
	FunctionPtr creator;
	// ����
	int generation;

	// �R���X�g���N�^
	Variable(const NdArrayPtr& data, const std::string& name = "") :
		data(data),
		name(name),
		generation(0)
	{}

	// �f�X�g���N�^
	virtual ~Variable() {}

	// �������̊֐���ݒ�
	void set_creator(const FunctionPtr& func);

	// �t�`�d(�ċA)
	void backward(bool retain_grad = false);

	// ������������
	void cleargrad() {
		this->grad = nullptr;
	}

	// NdArray�ֈϏ����郁���o
	decltype(auto) shape() { return data->shape(); }
	decltype(auto) size() { return data->size(); }
};

extern std::ostream& operator<<(std::ostream& ost, const Variable& v);
extern std::ostream& operator<<(std::ostream& ost, const VariablePtr& p);

// �֐��N���X
class Function : public std::enable_shared_from_this<Function>
{
public:
	// ���̓f�[�^
	VariablePtrList inputs;
	// �o�̓f�[�^
	VariableWPtrList outputs;
	// ����
	int generation;

	// �f�X�g���N�^
	virtual ~Function() {}

	// ()���Z�q
	VariablePtrList operator()(const NdArrayPtr& input)
	{
		// VariantPtr�ɕϊ����ď���
		return (*this)(as_variable(input));
	}

	// ()���Z�q
	VariablePtrList operator()(const VariablePtr& input)
	{
		// ���X�g�ɕϊ����ď���
		return (*this)(VariablePtrList({ input }));
	}

	// ()���Z�q
	VariablePtrList operator()(const VariablePtrList& inputs)
	{
		// ���̓f�[�^����NdArray�����o��
		auto xs = NdArrayPtrList();
		for (const auto& i : inputs) {
			xs.push_back(i->data);
		}

		// ���`�d
		auto ys = this->forward(xs);

		// �v�Z���ʂ���o�̓f�[�^���쐬
		auto outputs = VariablePtrList();
		for (const auto& y : ys) {
			auto o = as_variable(as_array(*y));
			o->set_creator(shared_from_this());
			outputs.push_back(o);
		}

		// �t�`�d�\�̏ꍇ
		if (Config::get_instance().param["enable_backprop"]) {
			// ���̓f�[�^�̂����ő�l�̐�������g�̐���Ƃ���
			auto max_elem = std::max_element(
				inputs.cbegin(), inputs.cend(),
				[](VariablePtr lhs, VariablePtr rhs) { return lhs->generation < rhs->generation; }
			);
			this->generation = (*max_elem)->generation;

			// ���o�̓f�[�^��ێ�����
			this->inputs = inputs;
			this->outputs = VariableWPtrList();
			for (const auto& o : outputs) {
				VariableWPtr w = o;
				this->outputs.push_back(w);
			}
		}

		return outputs;
	}

	// ���`�d
	virtual NdArrayPtrList forward(const NdArrayPtrList& xs) = 0;
	// �t�`�d
	virtual NdArrayPtrList backward(const NdArrayPtrList& gy) = 0;
};

// �֐��N���X�i���Z�j
class Add : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 + x1;
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		return { gys[0], gys[0] };
	}
};

// �֐��N���X�i���Z�j
class Sub : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 - x1;
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto gy = *(gys[0]);
		return { as_array(gy), as_array(-gy) };
	}
};

// �֐��N���X�i��Z�j
class Mul : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 * x1;
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x0 = *(this->inputs[0]->data);
		auto x1 = *(this->inputs[1]->data);
		auto gy = *(gys[0]);
		return { as_array(gy * x1), as_array(gy * x0) };
	}
};

// �֐��N���X�i���Z�j
class Div : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x0 = *(xs[0]);
		auto x1 = *(xs[1]);
		auto y = x0 / x1;
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x0 = *(this->inputs[0]->data);
		auto x1 = *(this->inputs[1]->data);
		auto gy = *(gys[0]);
		auto gx0 = as_array(gy / x1);
		auto gx1 = as_array(gy * (-x0 / nc::power(x1, 2)));
		return { gx0, gx1 };
	}
};

// �֐��N���X�i�����j
class Pos : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		// ���͒l�����̂܂ܕԂ����������A���`�d�ł͐V�����C���X�^���X�ɂ���K�v������
		auto x = *(xs[0]);
		return { as_array(x) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		return gys;
	}
};

// �֐��N���X�i�����j
class Neg : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		return { as_array(-x) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto gy = *(gys[0]);
		return { as_array(-gy) };
	}
};

// �֐��N���X�i�ݏ�j
class Pow : public Function
{
public:
	uint32_t c;

	// �R���X�g���N�^
	Pow(uint32_t c) : c(c) {}

	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::power(x, this->c);
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x = *(this->inputs[0]->data);
		auto gy = *(gys[0]);
		auto c = this->c;
		auto gx = static_cast<data_t>(c) * nc::power(x, c - 1)  * gy;
		return { as_array(gx) };
	}
};

// �֐��N���X�i2��j
class Square : public Function
{
public:
	// ���`�d
	NdArrayPtrList forward(const NdArrayPtrList& xs) override
	{
		auto x = *(xs[0]);
		auto y = nc::power(x, 2);
		return { as_array(y) };
	}
	// �t�`�d
	NdArrayPtrList backward(const NdArrayPtrList& gys) override
	{
		auto x = *(this->inputs[0]->data);
		auto gy = *(gys[0]);
		auto gx = 2.0 * x * gy;
		return { as_array(gx) };
	}
};

//----------------------------------
// function
//----------------------------------

// ���Z
extern VariablePtr add(const VariablePtr& x0, const VariablePtr& x1);
// ���Z
extern VariablePtr sub(const VariablePtr& x0, const VariablePtr& x1);
// ��Z
extern VariablePtr mul(const VariablePtr& x0, const VariablePtr& x1);
// ���Z
extern VariablePtr div(const VariablePtr& x0, const VariablePtr& x1);
// ����
extern VariablePtr pos(const VariablePtr& x);
// ����
extern VariablePtr neg(const VariablePtr& x);
// �ݏ�
extern VariablePtr pow(const VariablePtr& x0, uint32_t c);
// 2��
extern VariablePtr square(const VariablePtr& x0);

// VariablePtr�̉��Z�q�I�[�o�[���[�h
// �񍀉��Z�q +
extern VariablePtr operator+(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator+(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator+(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator+(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator+(data_t lhs, const VariablePtr& rhs);
// �񍀉��Z�q -
extern VariablePtr operator-(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator-(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator-(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator-(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator-(data_t lhs, const VariablePtr& rhs);
// �񍀉��Z�q *
extern VariablePtr operator*(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator*(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator*(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator*(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator*(data_t lhs, const VariablePtr& rhs);
// �񍀉��Z�q /
extern VariablePtr operator/(const VariablePtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator/(const VariablePtr& lhs, const NdArrayPtr& rhs);
extern VariablePtr operator/(const NdArrayPtr& lhs, const VariablePtr& rhs);
extern VariablePtr operator/(const VariablePtr& lhs, data_t rhs);
extern VariablePtr operator/(data_t lhs, const VariablePtr& rhs);
// �P�����Z�q +
extern VariablePtr operator+(const VariablePtr& data);
// �P�����Z�q -
extern VariablePtr operator-(const VariablePtr& data);

}	// namespace dezerocpp
