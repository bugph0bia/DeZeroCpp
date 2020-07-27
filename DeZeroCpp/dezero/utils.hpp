#pragma once

#include "../dezero/dezero.hpp"

namespace dz
{

//----------------------------------
// General Utility
//----------------------------------

// 文字列の全置換
inline std::string replace_all(const std::string& target_str, const std::string& old_str, const std::string& new_str)
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

// 階乗計算
inline double factorial(uint32_t x)
{
	// 32bit整数ではオーバーフローするのでdoubleとする
	double y = 1.0;
	for (; x > 0; x--) {
		y *= static_cast<double>(x);
	}
	return y;
}

// NdArray用の broadcast_to
// ※NdArrayは行列に次元固定されているため、それを前提とした簡易処理とする
inline NdArray broadcast_to(const NdArray& in_array, const nc::Shape& shape)
{
	// ブロードキャスト可能かチェック
	//assert(in_array.shape().rows == 1 || in_array.shape().rows == shape.rows);
	//assert(in_array.shape().cols == 1 || in_array.shape().cols == shape.cols);

	NdArray out_array;

	// スカラーをブロードキャスト
	if (in_array.shape().rows == 1 && in_array.shape().cols == 1) {
		// スカラーの値で全要素を埋める
		out_array = NdArray(shape).fill(0);
		out_array.fill(in_array[0]);
	}
	// 行方向のブロードキャスト
	else if (in_array.shape().rows == 1) {
		// 入力データの行方向のベクトルを取得
		std::vector<data_t> row_vec = in_array.toStlVector();
		// ブロードキャストして行列に拡張
		std::vector<std::vector<data_t>> mat;
		for (uint32_t i = 0; i < shape.rows; i++) mat.push_back(row_vec);
		out_array = NdArray(mat);
	}
	// 列方向のブロードキャスト
	else if (in_array.shape().rows == 1) {
		// 入力データの列方向のベクトルを取得
		std::vector<data_t> col_vec = in_array.transpose().toStlVector();
		// 行方向にブロードキャストして行列に拡張してから転地する
		std::vector<std::vector<data_t>> mat;
		for (uint32_t i = 0; i < shape.cols; i++) mat.push_back(col_vec);
		out_array = NdArray(mat).transpose();
	}
	else {
		// ブロードキャスト不可（不要）の場合は変換しない
		out_array = in_array;
	}

	return out_array;
}

// NdArray用の sum_to
// ※NdArrayは行列に次元固定されているためその前提の処理とする
inline NdArray sum_to(const NdArray& in_array, const nc::Shape& shape)
{
	// 計算可能かチェック
	assert(shape.rows == 1 || in_array.shape().rows == shape.rows);
	assert(shape.cols == 1 || in_array.shape().cols == shape.cols);

	NdArray out_array;

	// スカラーへ合計
	if (shape.rows == 1 && shape.cols == 1) {
		out_array = in_array.sum();
	}
	// 行方向の合計
	else if (shape.rows == 1) {
		out_array = NdArray(shape).fill(0);
		// 全行ループ
		for (uint32_t r = 0; r < in_array.shape().rows; r++) {
			// 1行ずつ加算する
			out_array += in_array(r, in_array.cSlice());
		}
	}
	// 列方向の合計
	else if (shape.cols == 1) {
		out_array = NdArray(shape).fill(0);
		// 全行ループ
		for (uint32_t c = 0; c < in_array.shape().cols; c++) {
			// 1列ずつ加算する
			out_array += in_array(in_array.rSlice(), c);
		}
	}
	else {
		out_array = in_array;
	}

	return out_array;
}

// 2つの NdArray を相互的にブロードキャストする
// ※NdArrayは四則演算の際などに自動的にブロードキャストが行われないため明示的にこの関数を利用する
inline void broadcast_mutual(NdArray& a0, NdArray& a1)
{
	auto a0_shape = a0.shape();
	auto a1_shape = a1.shape();
	a0 = broadcast_to(a0, a1_shape);
	a1 = broadcast_to(a1, a0_shape);
}

//----------------------------------
// DOT Language
//----------------------------------

// オブジェクトのID（アドレス値）を取得
template<typename T>
static uintptr_t id(const std::shared_ptr<T>& d)
{
	// VariablePtr, FunctionPtr のどちらかを想定するため
	// std::shared_ptr 型のみ使用できるように制限
	return reinterpret_cast<uintptr_t>(&*d);
}

// VariableをDOT出力
static std::string dot_var(const VariablePtr& v, bool verbose = false)
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

// FunctionをDOT出力
static std::string dot_func(const FunctionPtr& f)
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

// DOTグラフを出力
inline std::string get_dot_graph(const VariablePtr& output, bool verbose = true)
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

// DOTグラフから指定ファイルにレンダリング
inline void plot_dot_graph(const VariablePtr& output, bool verbose = true, const std::string& to_file = "graph.png")
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

}	// namespace dz
