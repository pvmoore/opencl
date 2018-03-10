#pragma once

namespace opencl {

class Platform {
	vector<Device> devices;
public:
	cl_platform_id id;
	string name;
	string vendor;
	string version;
	string profile;
	string extensions;

	Platform(cl_platform_id id) : id(id) { 
		queryPlatformInfo(); 
		queryDevices(); 
	}

	Context createContext(cl_device_type type, vector<cl_context_properties> props = {}) {
		props.insert(props.begin(), (cl_context_properties)id);
		props.insert(props.begin(), CL_CONTEXT_PLATFORM);
		props.push_back(0);

		/// Select the first matching device
		int deviceIndex = -1;
		for(int i = 0; i<devices.size(); i++) {
			if(devices[i].type == type) {
				deviceIndex = i;
				break;
			}
		}
		if(deviceIndex == -1) throw std::runtime_error("Can't find OpenCL device on this platform");

		cl_device_id deviceIds[] = {devices[deviceIndex].id};
		int err;
		cl_context contextId = clCreateContext(
			props.data(),
			1,
			deviceIds,
			nullptr,
			nullptr,
			&err
		);
		throwOnCLError(err);
		return Context{contextId, devices[deviceIndex]};
	}
	string toString() const {
		CharBuffer buf{"Platform {\n"};

		buf.append("Name       : ").append(name).append("\n");
		buf.append("Vendor     : ").append(vendor).append("\n");
		buf.append("Version    : ").append(version).append("\n");
		buf.append("Profile    : ").append(profile).append("\n");
		buf.append("Extensions : ").append(extensions).append("\n");

		buf.appendFmt("\nDevices: (%u)\n\n", devices.size());

		for(auto& dev : devices) {
			buf.append(dev.toString());
			buf.append("\n");
		}
		buf.append("}");
		return buf.std_str();
	}
private:
	void queryDevices() {
		uint numDevices;
		cl_device_id* deviceIDs;

		throwOnCLError(clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
		deviceIDs = new cl_device_id[numDevices];
		throwOnCLError(clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, numDevices, deviceIDs, &numDevices));
		
		for(uint i = 0; i < numDevices; ++i) {
			devices.push_back(Device{deviceIDs[i]});
		}
		delete[] deviceIDs;
	}
	void queryPlatformInfo() {
		char buf[1024];
		throwOnCLError(clGetPlatformInfo(id, CL_PLATFORM_NAME, sizeof(buf), &buf, nullptr));
		name += buf;
		throwOnCLError(clGetPlatformInfo(id, CL_PLATFORM_VENDOR, sizeof(buf), &buf, nullptr));
		vendor += buf;
		throwOnCLError(clGetPlatformInfo(id, CL_PLATFORM_VERSION, sizeof(buf), &buf, nullptr));
		version += buf;
		throwOnCLError(clGetPlatformInfo(id, CL_PLATFORM_PROFILE, sizeof(buf), &buf, nullptr));
		profile += buf;
		throwOnCLError(clGetPlatformInfo(id, CL_PLATFORM_EXTENSIONS, sizeof(buf), &buf, nullptr));
		extensions += buf;
	}
};

} /// opencl
