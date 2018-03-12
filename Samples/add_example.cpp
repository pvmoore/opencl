#include "_pch.h"

using namespace core;
using std::string;
using std::wstring;
using std::vector;
using std::shared_ptr;

#include "../OpenCL/_exports.h"
using namespace opencl;

/// A simple 1D kernel.
/// Add two buffers and output to a third buffer.
void addExample() {
	printf("==========================\n");
	printf(" Running Add Kernel\n");
	printf("==========================\n\n");
	const uint N = 1024 * 1024;
	uint* inputA = nullptr;
	uint* inputB = nullptr;
	uint* output = nullptr;
	try{
		auto start = std::chrono::high_resolution_clock::now();

		OpenCL cl;
		auto platform = cl.createPlatform(CL_DEVICE_TYPE_GPU);
		auto context  = platform.createContext(CL_DEVICE_TYPE_GPU);
		auto queue    = context.createQueue(true);

		/// Create some data
		inputA = new uint[N];
		inputB = new uint[N];
		output = new uint[N];

		for(uint i = 0; i < N; i++) {
			inputA[i] = i;
			inputB[i] = i;
			output[i] = 0; // zero the output too
		}

		Buffer inputBuffer1 = context.createDeviceBuffer(sizeof(uint) * N, CL_MEM_READ_ONLY);
		Buffer inputBuffer2 = context.createDeviceBuffer(sizeof(uint) * N, CL_MEM_READ_ONLY);
		Buffer outputBuffer = context.createDeviceBuffer(sizeof(uint) * N, CL_MEM_WRITE_ONLY);

		vector<string> options = {"-Werror", "-I MyIncludes/", "-D MYDEF=2", "-O5"};
		Program program = context.createProgram(L"Kernels/add.cl", options);

		uint delta = 50;

		Kernel kernel = program.getKernel("Add");
		kernel.setArg(0, inputBuffer1);
		kernel.setArg(1, inputBuffer2);
		kernel.setArg(2, outputBuffer);
		kernel.setArg(3, sizeof(uint), (void*)&delta);

		/// Write our input buffers to the device
		queue.enqueueWriteBuffer(inputBuffer1, inputA);
		queue.enqueueWriteBuffer(inputBuffer2, inputB);

		/// Execute the kernel
		Event kernelEvent;
		queue.enqueueKernel(
			kernel,
			{N},					// global sizes
			{},						// local sizes
			{{}, &kernelEvent}		// events
		);

		/// Read back the data
		queue.enqueueReadBuffer(outputBuffer, output, CL_TRUE);

		queue.finish();
		auto end = std::chrono::high_resolution_clock::now();

		auto kernelTime = kernelEvent.getRunTime();
		kernelEvent.release();

		printf("\n");
		printf("Num kernel threads executed .. %u\n", N);
		printf("Total time ................... %.3f ms\n", (end - start).count() * 1e-6);
		printf("Kernel time .................. %.3f ms\n\n", kernelTime * 1e-6);

		/// Check the results
		for(int i = 0; i < N; i++) {
			assert(output[i] = i + i + delta);
		}

	} catch(std::exception& e) {
		printf("FAIL: %s\n", e.what());
	}
	if(inputA) delete[] inputA;
	if(inputB) delete[] inputB;
	if(output) delete[] output;
}
