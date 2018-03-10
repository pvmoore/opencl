// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

// Target Windows 7 and above          
#define WINVER		 _WIN32_WINNT_WIN7
#define _WIN32_WINNT _WIN32_WINNT_WIN7

#define NOMINMAX	
#define _ALLOW_RTCc_IN_STL

#include <cstdio>
#include <cassert>
#include <crtdbg.h>

/// std headers
#include <memory>
#include <string>
#include <vector>
#include <exception>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <random>

/// OpenCL
#include <CL/opencl.h>
#pragma comment(lib, "../External/OpenCL")

/// Core
#include <Core/Core/core.h>
#ifdef _DEBUG
#pragma comment(lib, "../../Core/x64/Debug/Core.lib")
#else 
#pragma comment(lib, "../../Core/x64/Release/Core.lib")
#endif

#pragma warning(disable : 4101) /// unused variable
