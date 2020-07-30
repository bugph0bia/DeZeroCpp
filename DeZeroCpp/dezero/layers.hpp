#pragma once

#include "../dezero/dezero.hpp"

namespace dz::layers
{

namespace F = functions;

//----------------------------------
// class
//----------------------------------

// �v���p�e�B��set/get�̑㗝�����N���X
class PropProxy
{
private:
	// �v���p�e�B
	std::unordered_map<std::string, VariablePtr>& props;
	// �v���p�e�B���̃p�����[�^�ꗗ
	std::set<std::string>& param_names;
	// �L�[
	const std::string& key;

public:
	// �R���X�g���N�^
	PropProxy(std::unordered_map<std::string, VariablePtr>& props, std::set<std::string>& param_names, const std::string& key) :
		props(props),
		param_names(param_names),
		key(key)
	{}

	// �R�s�[/���[�u�R���X�g���N�^
	PropProxy(const PropProxy&) = default;
	PropProxy(PropProxy&&) = default;

	// �R�s�[������Z�q (set�㗝)
	PropProxy& operator=(const VariablePtr& value)
	{
		if (typeid(*value) == typeid(Parameter)) {
			param_names.insert(key);
		}
		props[key] = value;
		return *this;
	}

	// VariablePtr�ւ̃L���X�g���Z�q (get�㗝)
	operator VariablePtr() const noexcept
	{
		auto v = VariablePtr();
		if (props.find(key) != props.end())
			v = props[key];
		else
			assert(false);	// ���݂��Ȃ��v���p�e�B�̎Q��

		return v;
	}

	// �A���[���Z�q�őΏۂ� VariablePtr �̃����o�𒼐ڑ���
	VariablePtr operator->() const noexcept
	{
		return static_cast<VariablePtr>(*this);
	}
};

class Layer
{
protected:
	using props_type = std::unordered_map<std::string, VariablePtr>;
	using params_type = std::set<std::string>;

	// �v���p�e�B
	props_type props;
	// �v���p�e�B���̃p�����[�^�ꗗ
	params_type param_names;

	// ���̓f�[�^
	VariableWPtrList inputs;
	// �o�̓f�[�^
	VariableWPtrList outputs;

public:
	// �f�X�g���N�^
	virtual ~Layer() {}

	// �v���p�e�B��set/get
	// []���Z�q���� this �ƕ��p����Ƃ��ɔώG�ɂȂ�̂ł�������g�p����Ɨǂ�
	PropProxy prop(const std::string& key)
	{
		return (*this)[key];
	}

	// []���Z�q�F�v���p�e�B��set/get
	PropProxy operator[](const std::string& key)
	{
		return PropProxy(props, param_names, key);
	}

	// ()���Z�q
	VariablePtrList operator()(const NdArrayPtr& input)
	{
		// VariantPtr�ɕϊ����ď���
		return (*this)(as_variable(input));
	}
	VariablePtrList operator()(const VariablePtr& input)
	{
		// ���X�g�ɕϊ����ď���
		return (*this)(VariablePtrList({ input }));
	}
	VariablePtrList operator()(const VariablePtrList& inputs)
	{
		// ���`�d
		auto outputs = this->forward(inputs);

		// ���o�̓f�[�^��ێ�����
		this->inputs = VariableWPtrList();
		for (const auto& i : inputs) {
			VariableWPtr w = i;
			this->inputs.push_back(w);
		}
		this->outputs = VariableWPtrList();
		for (const auto& o : outputs) {
			VariableWPtr w = o;
			this->outputs.push_back(w);
		}

		return outputs;
	}

	// ���`�d
	virtual VariablePtrList forward(const VariablePtrList& xs) = 0;

	// �p�����[�^�̃R���N�V�����𐶐�
	decltype(auto) params()
	{
		// �v���p�e�B���p�����[�^�ł��邩���f
		auto is_param = [this](const props_type::value_type& kv)
		{
			return this->param_names.find(kv.first) != this->param_names.end();
		};

		// �v���p�e�B����p�����[�^�݂̂𒊏o
		props_type props_param_only;
		std::copy_if(this->props.begin(), this->props.end(), std::inserter(props_param_only, props_param_only.end()), is_param);

		// �v���p�e�B�̒l�̃R���N�V������Ԃ�
		std::set<VariablePtr> param_values;
		for (auto& kv : props_param_only) {
			param_values.insert(kv.second);
		}
		return param_values;
	}

	// �S�p�����[�^�̌��z��������
	void cleargrads()
	{
		// �p�����[�^�𒊏o���Č��z��������
		for (auto& p : this->params()) {
			p->cleargrad();
		}
	}
};

// ���C���N���X�i���`�ϊ�/�S�����j
class Linear : public Layer
{
public:
	// ���o�̓f�[�^�T�C�Y
	uint32_t in_size;
	uint32_t out_size;

	// �R���X�g���N�^
	Linear(uint32_t out_size, uint32_t in_size = 0, bool nobias = false) :
		in_size(in_size),
		out_size(out_size)
	{
		// �d�݂̏�����
		this->prop("W") = as_parameter(nullptr, "W");
		// in_size ���w�肳��Ă��Ȃ��ꍇ�͌��
		if (this->in_size != 0) {
			this->init_W();
		}

		// �o�C�A�X�̏�����
		if (!nobias) {
			this->prop("b") = as_parameter(as_array(nc::zeros<data_t>({ 1, out_size })), "b");
		}
	}

	// �d�݂̏�����
	void init_W()
	{
		auto I = this->in_size;
		auto O = this->out_size;
		auto W_data = nc::random::randN<data_t>({ I, O }) * nc::sqrt<data_t>(1.0 / I);
		this->prop("W")->data = as_array(W_data);
	}

	// ���`�d
	virtual VariablePtrList forward(const VariablePtrList& xs) override
	{
		auto x = xs[0];

		// �f�[�^�𗬂��^�C�~���O�ŏd�݂�������
		if (!this->prop("W")->data) {
			this->in_size = x->shape().cols;
			this->init_W();
		}
		auto y = F::linear(x, this->prop("W"), this->prop("b"));
		return { y };
	}
};

}	// namespace dz::layers
