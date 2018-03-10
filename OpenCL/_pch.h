#pragma once
        
// Target Windows 7 and above          
#define WINVER		 _WIN32_WINNT_WIN7
#define _WIN32_WINNT _WIN32_WINNT_WIN7

#define NOMINMAX		
#define _ALLOW_RTCc_IN_STL

//#include <cstdio>
//#include <cstdlib>
#include <cassert>
//#include <cstring>

/// std headers
#include <memory>
#include <string>
#include <vector>
#include <exception>
#include <algorithm>
#include <tuple>

/// OpenCL headers
#include <CL/opencl.h>

/// Core
#include <Core/Core/core.h>

#pragma warning(disable : 4101) /// unused variable