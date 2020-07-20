
#include "pch.h"

#include "utils.hpp"

namespace dz
{

std::string replace_all(const std::string& target_str, const std::string& old_str, const std::string& new_str)
{
	std::string result_str = target_str;
	std::string::size_type Pos(result_str.find(old_str));
	while (Pos != std::string::npos)
	{
		result_str.replace(Pos, old_str.length(), new_str);
		Pos = result_str.find(old_str, Pos + new_str.length());
	}
	return result_str;
}

template<typename T>
uintptr_t id(const std::shared_ptr<T>& d)
{
	// VariablePtr, FunctionPtr �̂ǂ��炩��z�肷�邽��
	// std::shared_ptr �^�̂ݎg�p�ł���悤�ɐ���
	return reinterpret_cast<uintptr_t>(&*d);
}

std::string dot_var(const VariablePtr& v, bool verbose = false)
{
	std::ostringstream osst;

	// �ϐ��̖���
	std::string name = v->name;

	if (verbose && v->data) {
		if (!v->name.empty()) {
			name += ": ";
		}

		// ���̂ɉ����āA�f�[�^�̌`��ƌ^���o��
		osst << name << v->shape() << " " << typeid(data_t).name();
		name = osst.str();
		name = replace_all(name, "\n", "");
		// osst���N���A
		osst.str("");
		osst.clear();
	}

	// �ϐ��m�[�h��dot�`���ŏo��
	osst << id(v) << " [label=\"" << name << "\", color=orange, style=filled]" << std::endl;
	return osst.str();
}

std::string dot_func(const FunctionPtr& f)
{
	std::ostringstream osst;
	std::string txt;

	// �N���X�����擾
	std::string class_name = typeid(*f).name();
	// ���O��ԂȂǂ̏A�E������ꍇ�́A�N���X���݂̂𒊏o
	auto cpos = class_name.find_last_of(":");
	if (cpos != std::string::npos) {
		cpos += 1;
		class_name = class_name.substr(cpos, class_name.size() - cpos);
	}

	// �֐��m�[�h��dot�`���ŏo��
	osst << id(f) << " [label=\"" << class_name << "\", color=lightblue, style=filled, shape=box]" << std::endl;
	txt = osst.str();
	// osst���N���A
	osst.str("");
	osst.clear();

	for (const auto& x : f->inputs) {
		// ���̓f�[�^�Ɗ֐��̃G�b�W��dot�`���ŏo��
		osst << id(x) << " -> " << id(f) << std::endl;
		txt += osst.str();
		// osst���N���A
		osst.str("");
		osst.clear();
	}
	for (const auto& y : f->outputs) {
		// �֐��Əo�̓f�[�^�̃G�b�W��dot�`���ŏo��
		osst << id(f) << " -> " << id(y.lock()) << std::endl;
		txt += osst.str();
		// osst���N���A
		osst.str("");
		osst.clear();
	}

	return txt;
}

std::string get_dot_graph(const VariablePtr& output, bool verbose /*=true*/)
{
	// dot�`���̕�����
	std::string txt;
	// �֐����X�g
	auto funcs = std::list<FunctionPtr>();
	// �����ς݊֐��Z�b�g
	auto seen_set = std::set<FunctionPtr>();

	// �N���[�W���F�֐����X�g�֒ǉ�
	auto add_func = [&funcs, &seen_set](const FunctionPtr& f) {
		// ���X�g�֖��ǉ��̊֐��Ȃ�
		if (seen_set.find(f) == seen_set.end()) {
			// ���X�g�֒ǉ�
			funcs.push_back(f);
			seen_set.insert(f);
		}
	};

	// �ŏ��̊֐������X�g�ɒǉ�
	add_func(output->creator);
	// �ŏ��̕ϐ���dot�`���ŏo��
	txt += dot_var(output, verbose);

	// �֐����X�g����ɂȂ�܂Ń��[�v
	while (!funcs.empty()) {
		// ���X�g����֐������o��
		auto f = funcs.back();
		funcs.pop_back();
		// �֐���dot�`���ŏo��
		txt += dot_func(f);

		// ���̓f�[�^�ƌ��z�̃y�A�����[�v
		for (const auto& x : f->inputs) {
			// �ϐ���dot�`���ŏo��
			txt += dot_var(x, verbose);

			// �P�O�̊֐������X�g�ɒǉ�
			if (x->creator) {
				add_func(x->creator);
			}
		}
	}

	// dot�`���̑S�̂𐮂��ďo��
	return "digraph g {\n" + txt + "}";
}

void plot_dot_graph(const VariablePtr& output, bool verbose /*=true*/, const std::string& to_file /*="graph.png"*/)
{
	// dot�`���̃f�[�^�𐶐�
	std::string dot_graph = get_dot_graph(output, verbose);

	// �o�̓t�@�C���̃p�X���
	std::filesystem::path p = to_file;

	// dot�t�@�C����ۑ�
	std::string graph_path = p.parent_path().string() + "tmp_graph.dot";	// �摜�Ɠ����ꏊ�ɏo��
	std::ofstream f(graph_path);
	f << dot_graph;
	f.close();

	// dot�R�}���h�����s
	auto ext = p.extension().string();
	ext = ext.substr(1);	// �擪�� '.' ���폜
	std::string cmd = "dot " + graph_path + " -T " + ext + " -o " + to_file;
	std::system(cmd.c_str());
}

}	//namespace dz
