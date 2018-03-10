#include "_pch.h"

using namespace core;
using std::string;
using std::vector;
using std::shared_ptr;

#include "../OpenCL/_exports.h"
using namespace opencl;

/// Enqueue a kernel from within another kernel. This is an OpenCL 2.0 feature. 
void enqueueExample() {
	printf("==========================\n");
	printf(" Running Enqueue Kernel\n");
	printf("==========================\n\n");
	const uint N   = 10;
	float* inData  = nullptr;
	float* outData = nullptr;
	try{
		auto start = std::chrono::high_resolution_clock::now();

		OpenCL cl;
		auto platform = cl.createPlatform(CL_DEVICE_TYPE_GPU);
		auto context  = platform.createContext(CL_DEVICE_TYPE_GPU);
		auto queue    = context.createQueue(true);

		inData = new float[N];
		outData = new float[N];
		for(int i=0; i<N; i++) {
			inData[i]  = (float)i;
			outData[i] = 0.0f;
		}

		auto inBuf = context.createDeviceBuffer(sizeof(float)*N,
			CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY);
		auto outBuf = context.createDeviceBuffer(sizeof(float)*N,
			CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY);

		queue.enqueueWriteBuffer(inBuf, inData);

		auto program = context.createProgram("Kernels/enqueue.cl", {
				"-g"		// adds debugging info about enqueued kernels
			}
		);

		auto kernel = program.getKernel("compute");
		kernel.setArg(0, inBuf);
		kernel.setArg(1, outBuf);

		/// Create the device queue otherwise you will get an out of resources error
		auto deviceQueue = context.createDeviceQueue();

		cl_event kernelEvent;
		queue.enqueueKernel(
			kernel,
			{N},					// global sizes
			{},						// local sizes
			{{}, &kernelEvent}		// events
		);	

		/// Blocking read
		queue.enqueueReadBuffer(outBuf, outData, CL_TRUE);

		queue.finish();
		auto end = std::chrono::high_resolution_clock::now();

		auto kernelTime = getRunTime(kernelEvent);
		release(kernelEvent);

		printf("\n");
		printf("Num kernel threads executed .. %u user + 3 device\n", N);
		printf("Total time ................... %.3f ms\n", (end - start).count() * 1e-6);
		printf("Kernel time .................. %.3f ms\n\n", kernelTime * 1e-6);

		/// Check the results
		for(int i = 0; i < N; i++) {
			assert(outData[i] == (float)i);
		}

	}catch(std::exception& e) {
		printf("FAIL: %s\n", e.what());
	}
	delete[] inData;
	delete[] outData;
}
