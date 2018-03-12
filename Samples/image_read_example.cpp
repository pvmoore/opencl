#include "_pch.h"

using namespace core;
using std::string;
using std::wstring;
using std::vector;
using std::shared_ptr;

#include "../OpenCL/_exports.h"
using namespace opencl;

/// Create an image and read random pixels from it.
void imageReadExample() {
	printf("==========================\n");
	printf(" Running Image Read Kernel\n");
	printf("==========================\n\n");

	const uint width  = 8192;
	const uint height = 8192;
	const uint N      = width * height;

	ubyte* input  = nullptr;
	ubyte* output = nullptr;
	try{
		auto start = std::chrono::high_resolution_clock::now();

		OpenCL cl;
		auto platform = cl.createPlatform(CL_DEVICE_TYPE_GPU);
		auto context  = platform.createContext(CL_DEVICE_TYPE_GPU);
		auto queue    = context.createQueue(true);

		input  = new ubyte[N];
		output = new ubyte[N];
		for(auto i = 0; i<N; i++) {
			input[i]  = 1;
			output[i] = 0;
		}

		auto outputBuf = context.createDeviceBuffer(sizeof(ubyte)*N, CL_MEM_WRITE_ONLY);
		auto image2d   = context.createDeviceImage(
			CL_MEM_READ_ONLY,
			cl_image_format{CL_R, CL_UNSIGNED_INT8},
			cl_image_desc{
				CL_MEM_OBJECT_IMAGE2D,  // type
				width,                  // width
				height,                 // height
				1,                      // depth
				1,                      // array size
				0,                      // row pitch
				0,                      // slice pitch
				0,                      // mipmap level
				0,                      // num samples
				nullptr                 // buffer/mem_object
			},
			nullptr
		);

		auto program = context.createProgram(L"kernels/image_read.cl");

		auto kernel = program.getKernel("RandomImageRead");
		kernel.setArg(0, image2d);
		kernel.setArg(1, outputBuf);

		queue.enqueueWriteImage(image2d, input);

		Event kernelEvent;
		queue.enqueueKernel(
			kernel,
			{N},					// global sizes
			{},						// local sizes
			{{}, &kernelEvent}		// events
		);

		/// Blocking read
		queue.enqueueReadBuffer(outputBuf, output, CL_TRUE);
		queue.finish();
		auto end = std::chrono::high_resolution_clock::now();

		auto kernelTime = kernelEvent.getRunTime();

		printf("\n");
		printf("Num kernel threads executed .. %u\n", N);
		printf("Total time ................... %.3f ms\n", (end - start).count() * 1e-6);
		printf("Kernel time .................. %.3f ms\n\n", kernelTime * 1e-6);

		/// Check the results
		for(int i = 0; i < N; i++) {
			assert(output[i] == 1);
		}

	}catch(std::exception& e) {
		printf("FAIL: %s\n", e.what());
	}
	delete[] input;
	delete[] output;
}