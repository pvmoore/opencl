#include "_pch.h"

using namespace core;
using std::string;
using std::vector;
using std::shared_ptr;

#include "_exports.h"

// Place globals and statics here

void throwOnCLError(int err) {
	if(err) {
		string msg;
		switch(err) {
			/// This is not a complete list
			case -4: msg = "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
			case -5: msg = "CL_OUT_OF_RESOURCES"; break;
			case -6: msg = "CL_OUT_OF_HOST_MEMORY"; break;
			case -7: msg = "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
			case -8: msg = "CL_MEM_COPY_OVERLAP"; break;
			case -11: msg = "CL_BUILD_PROGRAM_FAILURE"; break;
			case -12: msg = "CL_MAP_FAILURE"; break;
			case -14: msg = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
			case -30: msg = "CL_INVALID_VALUE"; break;
			case -34: msg = "CL_INVALID_CONTEXT"; break;
			case -37: msg = "CL_INVALID_HOST_PTR"; break;
			case -38: msg = "CL_INVALID_MEM_OBJECT"; break;
			case -40: msg = "CL_INVALID_IMAGE_SIZE"; break;
			case -43: msg = "CL_INVALID_BUILD_OPTIONS"; break;
			case -48: msg = "CL_INVALID_KERNEL"; break;
			case -49: msg = "CL_INVALID_ARG_INDEX"; break;
			case -50: msg = "CL_INVALID_ARG_VALUE"; break;
			case -51: msg = "CL_INVALID_ARG_SIZE"; break;
			case -52: msg = "CL_INVALID_KERNEL_ARGS"; break;
			case -54: msg = "CL_INVALID_WORK_GROUP_SIZE"; break;
			case -58: msg = "CL_INVALID_EVENT"; break;
			case -59: msg = "CL_INVALID_OPERATION"; break;
			case -60: msg = "CL_INVALID_GL_OBJECT"; break;
			default: msg = "UNKNOWN"; break;
		}
		string str = String::format("OpenCL error: %s (%d)", msg.c_str(), err);
		Log::write(str);
		throw std::runtime_error(str);
	}
}

namespace opencl {

void Kernel::createKernel() {
	int err;
	this->id = clCreateKernel(program.id, name.c_str(), &err);
	throwOnCLError(err);
}
void Kernel::getWorkGroupInfo(cl_kernel_work_group_info param, void* paramPtr, ulong paramSize) const {
	throwOnCLError(clGetKernelWorkGroupInfo(
		id,
		program.device.id,
		param,
		paramSize,
		paramPtr,
		nullptr
	));
}
std::tuple<uint, uint> Kernel::getSquareWorkGroupSize2D() const {
	uint n = (uint)getPreferredWorkGroupSizeMultiple();

	// AMD is always 64
	if(n == 64) return {8, 8};

	// NVidia and Intel seem to like 32
	if(n == 32) return {8, 4};

	// perfect square
	uint sq = (uint)sqrt((float)n);
	if(sq*sq == n) return {sq, sq};

	// find the innermost rectangle
	auto factors = Math::factorsOf(n);
	auto start   = 0;
	auto end     = factors.size() - 1;

	uint x=n, y=1;
	while(start<end) {
		x = factors[start];
		y = factors[end];
		start++;
		end--;
	}
	return std::make_tuple(x,y);
}

} /// opencl