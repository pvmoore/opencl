#include "_pch.h"

using namespace core;
using std::string;
using std::wstring;
using std::vector;
using std::shared_ptr;

#include "../OpenCL/_exports.h"
using namespace opencl;

/// Sort an array.
void sortExample() {
	printf("==========================\n");
	printf(" Running Sort Kernel\n");
	printf("==========================\n\n");

	const uint N = 1024*1024;
	float* randomData = nullptr;
	float* sortedData = nullptr;
	try{
		auto start = std::chrono::high_resolution_clock::now();

		OpenCL cl;
		auto platform = cl.createPlatform(CL_DEVICE_TYPE_GPU);
		auto context  = platform.createContext(CL_DEVICE_TYPE_GPU);
		auto queue    = context.createQueue(true);

		/// Create random data
		std::default_random_engine gen;
		std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
		randomData = new float[N];
		sortedData = new float[N];
		for(int i = 0; i < N; i++) {
			randomData[i] = uniform01(gen);
			sortedData[i] = 0.0f;
		}
		printf("First 64 values of random data:\n");
		for(int i = 0; i < 64; i++) {
			printf("%.2f ", randomData[i]);
		}
		printf("\n\n");

		/// Use mapped buffers
		auto inBuf  = context.createDeviceBuffer(sizeof(float)*N, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, randomData);
		auto outBuf = context.createDeviceBuffer(sizeof(float)*N, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sortedData);

		/// Map the inBuf and let the GPU read the randomData
		void* ptr = queue.enqueueMapBuffer(inBuf, 0, inBuf.size, CL_MAP_READ, CL_FALSE);
		assert(ptr==randomData);
		queue.enqueueUnmapMemObject(inBuf, ptr);

		bool ascending = true;

		uint WORK_GROUP_SIZE = (uint)context.device.maxWorkGroupSize;

		/// The data set needs to be a multiple of the work group size
		assert(N%WORK_GROUP_SIZE==0);

		auto program = context.createProgram(L"Kernels/sort.cl",
			{
				String::format("-D WORK_GROUP_SIZE=%u", WORK_GROUP_SIZE).c_str(),
				String::format("-D ASCENDING=%s", ascending ? "true":"false").c_str()
			}
		);

		auto sortKernel = program.getKernel("bitonicSortLocal");
		sortKernel.setArg(0, inBuf);

		auto mergeKernel = program.getKernel("merge");
		mergeKernel.setArg(0, inBuf);
		mergeKernel.setArg(1, outBuf);

		printf("\n");
		printf("maxWorkGroupSize ................ %llu\n", sortKernel.getMaxWorkGroupSize());
		printf("localMemSize .................... %llu\n", sortKernel.getLocalMemSize());
		printf("privateMemSize .................. %llu\n", sortKernel.getPrivateMemSize());
		printf("preferredWorkGroupSizeMultiple .. %llu\n", sortKernel.getPreferredWorkGroupSizeMultiple());

		/// Run the sortKernel
		printf("\nSorting %u local chunks of %u values\n", N/WORK_GROUP_SIZE, WORK_GROUP_SIZE);
		Event sortKernelEvent;
		queue.enqueueKernel(
			sortKernel,
			{N},					// global sizes
			{WORK_GROUP_SIZE},		// local sizes
			{{}, &sortKernelEvent}	// events
		);
		queue.enqueueBarrier();

		/// We now have lots of regions of locally sorted data. 
		/// We now need to run the merge kernel iteratively to merge the sorted chunks
		/// into larger sorted chunks until we have sorted the whole data set.

		/// eg. if the work group size = 4 and N = 16 we might have:
		///   0,3,5,7,  2,3,8,10,  1,5,8,10,  2,2,4,9
		/// We now run merge with a chunk size of 4.
		/// This will merge the 4-value sorted chunks into 8-value sorted chunks.
		///   0,2,3,3,5,7,8,10,	   1,2,2,4,5,8,9,10, 
		/// We run merge again with chunk size = 8 and the 8-value sorted chunks are merged into 
		/// one 16-value sorted chunk and we are finished:
		///   0,1,2,2,2,3,3,4,5,5,7,8,8,9,10,10

		/// Run the merge kernel iteratively
		uint chunkSize = WORK_GROUP_SIZE;
		while(chunkSize < N) {
			printf("Merging %u-value chunks -> %u value chunks\n", chunkSize, chunkSize*2);
			mergeKernel.setArg(2, chunkSize);
			queue.enqueueKernel(mergeKernel, {N});

			/// Copy the out buf back to in buf ready for the next round
			queue.enqueueCopyBuffer(outBuf, inBuf);

			chunkSize <<= 1;
		}
		/// Map the inBuf and write the random data
		void* ptr2 = queue.enqueueMapBuffer(inBuf, 0, inBuf.size, CL_MAP_WRITE, CL_FALSE);
		queue.enqueueUnmapMemObject(inBuf, ptr2);

		queue.finish();
		auto end = std::chrono::high_resolution_clock::now();

		auto sortKernelTime = sortKernelEvent.getRunTime();

		printf("\n");
		printf("Num kernel threads executed .. %u\n", N);
		printf("Total time ................... %.3f ms\n", (end - start).count() * 1e-6);
		printf("Sort kernel time ............. %.3f ms\n\n", sortKernelTime * 1e-6);

		/// Check that the results are sorted
		vector<float> values;
		values.resize(N);
		for(int i = 0; i < N; i++) {
			values[i] = randomData[i];
		}
		bool b = std::is_sorted(values.begin(), values.end());
		printf("is_sorted = %s\n\n", b ? "true":"false");

		printf("First 64 values of sorted data:\n");
		for(int i = 0; i < 64; i++) {
			printf("%.3f ", randomData[i]);
		}
		printf("\n\nLast 64 values of sorted data:\n");
		for(int i = 0; i < 64; i++) {
			printf("%.3f ", randomData[(N-1)-i]);
		}
		printf("\n");

	}catch(std::exception& e) {
		printf("FAIL: %s\n", e.what());
	}
	delete[] randomData;
	delete[] sortedData;
}