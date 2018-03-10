#pragma once

namespace opencl {

class MemObject {
public:
	cl_mem id;
	cl_mem_flags flags;
	
	MemObject(cl_mem id, cl_mem_flags flags) : id(id), flags(flags) {}
	~MemObject() {
		clReleaseMemObject(id);
	}
};

class Buffer final : public MemObject {
public:
	size_t size;
	Buffer(cl_mem id, cl_mem_flags flags, size_t sizeBytes) : MemObject(id,flags), size(sizeBytes) {}
};

class Image final : public MemObject {
public:
	ulong width, height;
	Image(cl_mem id, cl_mem_flags flags, ulong width, ulong height)
		: MemObject(id, flags), width(width), height(height) {}
};

} /// opencl