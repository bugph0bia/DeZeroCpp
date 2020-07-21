#pragma once

#define IS_SIMPLE_CORE

#include "utils.hpp"

#ifdef IS_SIMPLE_CORE
#include "core_simple.hpp"
#else
#include "core.hpp"
#endif	// #ifdef IS_SIMPLE_CORE


