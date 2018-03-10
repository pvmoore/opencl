#pragma once

namespace opencl {

class Kernel {
public:
	class Program& program;
	cl_kernel id;
	string name;

	Kernel(Program& program, const string& name) : program(program), name(name) { 
		createKernel(); 
	}
	~Kernel() { 
		clReleaseKernel(id); 
	}

	void setArg(uint index, const MemObject& mem) {
		setArg(index, sizeof(cl_mem), &mem.id);
	}
	void setArg(uint index, uint value) {
		setArg(index, sizeof(uint), &value);
	}
	void setArg(uint index, float value) {
		setArg(index, sizeof(float), &value);
	}
	void setArg(uint index, ulong size, const void* value) const {
		throwOnCLError(clSetKernelArg(id, index, size, value));
	}
	ulong getMaxWorkGroupSize() const {
		return getUlongWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE);
	}
	ulong getLocalMemSize() const {
		return getUlongWorkGroupInfo(CL_KERNEL_LOCAL_MEM_SIZE);
	}
	ulong getPrivateMemSize() const {
		return getUlongWorkGroupInfo(CL_KERNEL_PRIVATE_MEM_SIZE);
	}
	ulong getPreferredWorkGroupSizeMultiple() const {
		return getUlongWorkGroupInfo(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
	}
	std::tuple<uint, uint> getSquareWorkGroupSize2D() const;
private:
	ulong getUlongWorkGroupInfo(cl_kernel_work_group_info param) const {
		ulong value;
		getWorkGroupInfo(param, &value, sizeof(ulong));
		return value;
	}
	void createKernel();
	void getWorkGroupInfo(cl_kernel_work_group_info param, void* paramPtr, ulong paramSize) const;
};

} /// opencl