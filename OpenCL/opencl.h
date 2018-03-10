#pragma once

namespace opencl {

class OpenCL {
	vector<cl_platform_id> platformIds;
public:
	OpenCL() { 
		enumeratePlatforms();
	}

	uint numPlatforms() const { return (uint)platformIds.size(); }

	Platform createPlatform(cl_device_type type) const {
		for(auto& pid : platformIds) {
			if(canCreatePlatformWithDeviceType(pid, type)) return Platform{pid};
		}
		throw std::runtime_error("Unable to find OpenCL platform");
	}
private:
	void enumeratePlatforms() {
		uint numPlatforms;
		throwOnCLError(clGetPlatformIDs(0, nullptr, &numPlatforms));

		platformIds.resize(numPlatforms);

		throwOnCLError(clGetPlatformIDs(numPlatforms, platformIds.data(), nullptr));
	}
	bool canCreatePlatformWithDeviceType(cl_platform_id platformId, cl_device_type type) const {
		cl_context_properties props[] = {
			(cl_context_properties)CL_CONTEXT_PLATFORM,
			(cl_context_properties)platformId,
			(cl_context_properties)0
		};
		int err;
		auto ctx = clCreateContextFromType(
			props,
			type,
			nullptr,
			nullptr,
			&err);
		bool result = err == 0;
		if(ctx) clReleaseContext(ctx);
		return result;
	}
};

} /// opencl