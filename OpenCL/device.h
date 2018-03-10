#pragma once

namespace opencl {

class Device {
public:
	enum VendorID { UNKNOWN, NVIDIA, ATI, INTEL };

	cl_device_id id;
	VendorID vendorId = VendorID::UNKNOWN;
	cl_device_type type;
	uint maxComputeUnits;
	uint maxWorkItemDims;
	ulong maxWorkGroupSize;
	vector<ulong> maxWorkItemSizes;
	uint maxClockFreq;
	uint addressBits;
	uint maxConstantArgs;
	ulong maxMemAllocSize;
	ulong globalMemSize;
	ulong localMemSize;
	ulong maxConstantBufferSize;
	cl_bool available;
	cl_bool compilerAvailable;
	cl_bool littleEndian;
	cl_bool errorCorrection;
	cl_command_queue_properties queueProperties;
	cl_device_exec_capabilities execCaps;
	cl_device_fp_config fpConfig;
	ulong timerResolution;
	string name;
	string deviceVersion;
	string driverVersion;
	string extensions;

	// nvidia specific
	uint nvComputeCapabilityMajor, nvComputeCapabilityMinor, nvRegsPerBlock, nvWarpSize;
	cl_bool nvGpuOverlap, nvExecTimeout, nvIntegratedMemory;

	Device(cl_device_id id) : id(id) { 
		query(); 
	}

	string toString() const {
		CharBuffer buf;

		buf.append("Type                : ");
		switch(type) {
			case CL_DEVICE_TYPE_DEFAULT: buf.append("DEFAULT"); break;
			case CL_DEVICE_TYPE_CPU: buf.append("CPU"); break;
			case CL_DEVICE_TYPE_GPU: buf.append("GPU"); break;
			case CL_DEVICE_TYPE_ACCELERATOR: buf.append("ACCELERATOR"); break;
			case CL_DEVICE_TYPE_ALL: buf.append("ALL"); break;
		}	
		buf.append("\n");

		buf.append("Name                : ").append(name).append("\n");
		buf.append("Vendor              : ").append(vendorId==NVIDIA?"NVIDIA":
													vendorId==ATI?"ATI":
													vendorId==INTEL?"INTEL":
													"UNKNOWN").append("\n");
		buf.append("Available?          : ").append(available?"yes":"no").append("\n");
		buf.append("Device ver          : ").append(deviceVersion).append("\n");
		buf.append("Driver ver          : ").append(driverVersion).append("\n");
		buf.appendFmt("Max compute units   : %u\n", maxComputeUnits);
		buf.appendFmt("Max work item dims  : %u\n", maxWorkItemDims);
		buf.append("Max work item sizes : ");
		for(auto i=0; i<maxWorkItemSizes.size(); i++) {
			if(i>0) buf.append(" "); 
			buf.appendFmt("%llu", maxWorkItemSizes[i]);
		}
		buf.append("\n");

		buf.appendFmt("Max work group size : %llu\n", maxWorkGroupSize);
		buf.appendFmt("Max clock frequency : %u\n", maxClockFreq);
		buf.appendFmt("Max mem alloc size  : %llu MBs\n", maxMemAllocSize/(1024*1024));
		buf.appendFmt("Global mem size     : %llu MBs\n", globalMemSize/(1024*1024));
		buf.appendFmt("Local mem size      : %llu KBs\n",localMemSize/1024);
		buf.appendFmt("Max const buf size  : %llu KBs\n", maxConstantBufferSize/1024);
		buf.appendFmt("Max const args      : %u\n", maxConstantArgs);

		buf.append("Compiler available? : ").append(compilerAvailable?"yes":"no").append("\n");
		buf.append("Little endian?      : ").append(littleEndian?"yes":"no").append("\n");
		buf.append("Error correction?   : ").append(errorCorrection?"yes":"no").append("\n");

		buf.appendFmt("Profiling timer res : %llu nanoseconds\n", timerResolution);
		buf.append("Extensions          : ").append(extensions).append("\n");

		buf.append("Queue properties    : ");
		if(queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) buf.append("CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ");
		if(queueProperties & CL_QUEUE_PROFILING_ENABLE) buf.append("CL_QUEUE_PROFILING_ENABLE");
		buf.append("\n");

		buf.append("Execution caps	    : ");
		if(execCaps & CL_EXEC_KERNEL) buf.append("CL_EXEC_KERNEL ");
		if(execCaps & CL_EXEC_NATIVE_KERNEL) buf.append("CL_EXEC_NATIVE_KERNEL");
		buf.append("\n");

		buf.append("Single FP config    : ");
		if(fpConfig & CL_FP_DENORM) buf.append("CL_FP_DENORM ");
		if(fpConfig & CL_FP_INF_NAN) buf.append("CL_FP_INF_NAN ");
		if(fpConfig & CL_FP_ROUND_TO_NEAREST) buf.append("CL_FP_ROUND_TO_NEAREST ");
		if(fpConfig & CL_FP_ROUND_TO_ZERO) buf.append("CL_FP_ROUND_TO_ZERO ");
		if(fpConfig & CL_FP_ROUND_TO_INF) buf.append("CL_FP_ROUND_TO_INF ");
		if(fpConfig & CL_FP_FMA) buf.append("CL_FP_FMA ");
		buf.append("\n");

		if(vendorId==NVIDIA) {
		//	buf.append("nVidia compute capability : ").append(nvComputeCapabilityMajor).append(".").append(nvComputeCapabilityMinor).append("\n");
		//	buf.append("nVidia regs per block     : ").append(nvRegsPerBlock).append("\n");
		//	buf.append("nVidia warp size          : ").append(nvWarpSize).append("\n");

		//	buf.append("nVidia GPU overlap?       : ").append(nvGpuOverlap?"yes":"no").append("\n");
		//	buf.append("nVidia exec timeout?      : ").append(nvExecTimeout?"yes":"no").append("\n");
		//	buf.append("nVidia integrated memory? : ").append(nvIntegratedMemory?"yes":"no").append("\n");
		}
		return buf.std_str();
	}
private:
	void query() {
		char buf[1024];

		/// determine vendor
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(buf), &buf, nullptr));
		string cb = buf;
		if(cb.find("NVIDIA") != string::npos) {
			vendorId = VendorID::NVIDIA;
		} else if(cb.find("ATI") != string::npos ||
					cb.find("Advanced Micro Devices") != string::npos) {
			vendorId = VendorID::ATI;
		} else if(cb.find("Intel") != string::npos) {
			vendorId = VendorID::INTEL;
		} else {
			vendorId = VendorID::UNKNOWN;
			printf("Unknown vendor '%s'\n", cb.c_str());
		}

		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(type), &type, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDims), &maxWorkItemDims, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr));

		ulong* workitem_size = new ulong[maxWorkItemDims];
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDims * sizeof(ulong), workitem_size, nullptr));
		for(uint i = 0; i<maxWorkItemDims; i++) maxWorkItemSizes.push_back(workitem_size[i]);
		delete[] workitem_size;

		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFreq), &maxClockFreq, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_ADDRESS_BITS, sizeof(addressBits), &addressBits, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(maxConstantBufferSize), &maxConstantBufferSize, nullptr));

		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_AVAILABLE, sizeof(available), &available, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compilerAvailable), &compilerAvailable, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_ENDIAN_LITTLE, sizeof(littleEndian), &littleEndian, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(errorCorrection), &errorCorrection, nullptr));

		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_EXTENSIONS, sizeof(buf), &buf, nullptr));
		extensions += buf;

		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(buf), &buf, nullptr));
		name += buf;
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_VERSION, sizeof(buf), &buf, nullptr));
		deviceVersion += buf;
		throwOnCLError(clGetDeviceInfo(id, CL_DRIVER_VERSION, sizeof(buf), &buf, nullptr));
		driverVersion += buf;

		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queueProperties), &queueProperties, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(execCaps), &execCaps, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(timerResolution), &timerResolution, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(maxConstantArgs), &maxConstantArgs, nullptr));
		throwOnCLError(clGetDeviceInfo(id, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(fpConfig), &fpConfig, nullptr));

		if(vendorId==NVIDIA) {
		//	clGetDeviceInfo(device, CL_NV_DEVICE_COMPUTE_CAPABILITY_MAJOR, sizeof(nvComputeCapabilityMajor), &nvComputeCapabilityMajor, nullptr);
		//	clGetDeviceInfo(device, CL_NV_DEVICE_COMPUTE_CAPABILITY_MINOR, sizeof(nvComputeCapabilityMinor), &nvComputeCapabilityMinor, nullptr);
		//	clGetDeviceInfo(device, CL_NV_DEVICE_REGISTERS_PER_BLOCK, sizeof(nvRegsPerBlock), &nvRegsPerBlock, nullptr);
		//	clGetDeviceInfo(device, CL_NV_DEVICE_WARP_SIZE, sizeof(nvWarpSize), &nvWarpSize, nullptr);
		//	clGetDeviceInfo(device, CL_NV_DEVICE_GPU_OVERLAP, sizeof(nvGpuOverlap), &nvGpuOverlap, nullptr);
		//	clGetDeviceInfo(device, CL_NV_DEVICE_KERNEL_EXEC_TIMEOUT, sizeof(nvExecTimeout), &nvExecTimeout, nullptr);
		//	clGetDeviceInfo(device, CL_NV_DEVICE_INTEGRATED_MEMORY, sizeof(nvIntegratedMemory), &nvIntegratedMemory, nullptr);
		}
	}
};

} /// opencl