#pragma once

namespace opencl {

class Program {
	cl_context contextId;
public:
	cl_program id;
	string filename;
	Device& device;

	Program(cl_context ctxId, Device& device, const string& fileName, vector<string> options) 
		: contextId(ctxId), device(device), filename(fileName) 
	{
		load(options);
	}
	~Program() { 
		clReleaseProgram(id);
	}

	Kernel getKernel(const string& funcName) {
		return Kernel{*this, funcName};
	}
private:
	void load(vector<string> options) {
		string src = File::readText(filename);

		cl_int err;
		const char* srcStr = src.c_str();
		const size_t len = src.length();
		this->id = clCreateProgramWithSource(contextId, 1, (const char**)&srcStr, &len, &err);
		throwOnCLError(err);
		printf("Building program: %s\n", filename.c_str());

		/// Concatenate options
		string optionsStr;
		for(auto& it : options) {
			optionsStr += it + " ";
		}
		
		string standardOptions = 
			//"-Werror " 			            // Make all warnings into errors
			//"-w"                              // inhibit all warnings
			//"-cl-uniform-work-group-size "    // 2.0 only - requires that the global work-size be a multiple of the work-group size
			//"-O5 "
			"-cl-single-precision-constant " 	// Treat double precision floating-point constant as single precision constant
			"-cl-fast-relaxed-math "			// Enable all unsafe maths optimisations
			"-cl-mad-enable "
			"-cl-no-signed-zeros "
			"-cl-denorms-are-zero "
			"-cl-std=CL2.0 ";
		optionsStr += standardOptions;
		printf("Using options: %s\n", optionsStr.c_str());

		cl_device_id devices[] = {device.id};
		err = clBuildProgram(id, 1, devices, optionsStr.c_str(), nullptr, nullptr);
		if(err != CL_SUCCESS) {
			ulong sizeGiven = 0;
			char buildLog[10240] = {};
			clGetProgramBuildInfo(id, device.id, CL_PROGRAM_BUILD_LOG, _countof(buildLog), buildLog, &sizeGiven);

			string msg{buildLog, sizeGiven};
			throw std::runtime_error(("Compilation failed: " + msg).c_str());
		}
	}
};

} /// opencl