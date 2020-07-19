#pragma once

#define IS_SIMPLE_CORE

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
#include <map>
#include "NumCpp.hpp"

#ifdef IS_SIMPLE_CORE
#include "core_simple.hpp"
#else
#include "core.hpp"
#endif	// #ifdef IS_SIMPLE_CORE

#include "utils.hpp"

