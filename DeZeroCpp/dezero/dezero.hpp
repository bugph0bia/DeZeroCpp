#pragma once

#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <list>
#include <vector>
#include <set>
#include <unordered_map>
#include <variant>
#include <functional>

#include "NumCpp.hpp"

//#define IS_SIMPLE_CORE
#ifdef IS_SIMPLE_CORE
#include "core_simple.hpp"
#else
#include "core.hpp"
#endif	// #ifdef IS_SIMPLE_CORE

#include "functions.hpp"
#include "layers.hpp"
#include "models.hpp"
#include "utils.hpp"
#include "Optimizers.hpp"
