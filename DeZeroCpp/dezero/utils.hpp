#pragma once

#include "dezero.hpp"

namespace dz
{

extern std::string replace_all(const std::string& target_str, const std::string& old_str, const std::string& new_str);

extern std::string get_dot_graph(const VariablePtr& output, bool verbose = true);
extern void plot_dot_graph(const VariablePtr& output, bool verbose = true, const std::string& to_file = "graph.png");

extern uint32_t factorial(uint32_t x);










}	// namespace dz
