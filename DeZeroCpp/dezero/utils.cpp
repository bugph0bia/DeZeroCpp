
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
	// VariablePtr, FunctionPtr のどちらかを想定するため
	// std::shared_ptr 型のみ使用できるように制限
	return reinterpret_cast<uintptr_t>(&*d);
}

std::string dot_var(const VariablePtr& v, bool verbose = false)
{
	std::ostringstream osst;

	// 変数の名称
	std::string name = v->name;

	if (verbose && v->data) {
		if (!v->name.empty()) {
			name += ": ";
		}

		// 名称に加えて、データの形状と型も出力
		osst << name << v->shape() << " " << typeid(data_t).name();
		name = osst.str();
		name = replace_all(name, "\n", "");
		// osstをクリア
		osst.str("");
		osst.clear();
	}

	// 変数ノードをdot形式で出力
	osst << id(v) << " [label=\"" << name << "\", color=orange, style=filled]" << std::endl;
	return osst.str();
}

std::string dot_func(const FunctionPtr& f)
{
	std::ostringstream osst;
	std::string txt;

	// クラス名を取得
	std::string class_name = typeid(*f).name();
	// 名前空間などの就職がある場合は、クラス名のみを抽出
	auto cpos = class_name.find_last_of(":");
	if (cpos != std::string::npos) {
		cpos += 1;
		class_name = class_name.substr(cpos, class_name.size() - cpos);
	}

	// 関数ノードをdot形式で出力
	osst << id(f) << " [label=\"" << class_name << "\", color=lightblue, style=filled, shape=box]" << std::endl;
	txt = osst.str();
	// osstをクリア
	osst.str("");
	osst.clear();

	for (const auto& x : f->inputs) {
		// 入力データと関数のエッジをdot形式で出力
		osst << id(x) << " -> " << id(f) << std::endl;
		txt += osst.str();
		// osstをクリア
		osst.str("");
		osst.clear();
	}
	for (const auto& y : f->outputs) {
		// 関数と出力データのエッジをdot形式で出力
		osst << id(f) << " -> " << id(y.lock()) << std::endl;
		txt += osst.str();
		// osstをクリア
		osst.str("");
		osst.clear();
	}

	return txt;
}

std::string get_dot_graph(const VariablePtr& output, bool verbose /*=true*/)
{
	// dot形式の文字列
	std::string txt;
	// 関数リスト
	auto funcs = std::list<FunctionPtr>();
	// 処理済み関数セット
	auto seen_set = std::set<FunctionPtr>();

	// クロージャ：関数リストへ追加
	auto add_func = [&funcs, &seen_set](const FunctionPtr& f) {
		// リストへ未追加の関数なら
		if (seen_set.find(f) == seen_set.end()) {
			// リストへ追加
			funcs.push_back(f);
			seen_set.insert(f);
		}
	};

	// 最初の関数をリストに追加
	add_func(output->creator);
	// 最初の変数をdot形式で出力
	txt += dot_var(output, verbose);

	// 関数リストが空になるまでループ
	while (!funcs.empty()) {
		// リストから関数を取り出す
		auto f = funcs.back();
		funcs.pop_back();
		// 関数をdot形式で出力
		txt += dot_func(f);

		// 入力データと勾配のペアをループ
		for (const auto& x : f->inputs) {
			// 変数をdot形式で出力
			txt += dot_var(x, verbose);

			// １つ前の関数をリストに追加
			if (x->creator) {
				add_func(x->creator);
			}
		}
	}

	// dot形式の全体を整えて出力
	return "digraph g {\n" + txt + "}";
}

void plot_dot_graph(const VariablePtr& output, bool verbose /*=true*/, const std::string& to_file /*="graph.png"*/)
{
	// dot形式のデータを生成
	std::string dot_graph = get_dot_graph(output, verbose);

	// 出力ファイルのパス情報
	std::filesystem::path p = to_file;

	// dotファイルを保存
	std::string graph_path = p.parent_path().string() + "tmp_graph.dot";	// 画像と同じ場所に出力
	std::ofstream f(graph_path);
	f << dot_graph;
	f.close();

	// dotコマンドを実行
	auto ext = p.extension().string();
	ext = ext.substr(1);	// 先頭の '.' を削除
	std::string cmd = "dot " + graph_path + " -T " + ext + " -o " + to_file;
	std::system(cmd.c_str());
}

}	//namespace dz
