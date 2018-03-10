#pragma once

namespace opencl {

class CommandQueue {
public:
	struct EventArgs final {
		vector<cl_event> waitList;
		cl_event* event = nullptr;
		inline uint numWaitEvents() const { return (uint)waitList.size(); }
	};

	cl_command_queue id;

	CommandQueue(cl_command_queue id) : id(id) {}
	~CommandQueue() {
		if(id) clReleaseCommandQueue(id);
	}
	/// Read entire buffer
	void enqueueReadBuffer(const Buffer& buf, void* dest, cl_bool block, EventArgs args = {}) {
		enqueueReadBuffer(buf, dest, 0, buf.size, block, args);
	}
	/// Read partial buffer
	void enqueueReadBuffer(const Buffer& buf,
						   void* dest,
						   size_t readOffset,
						   size_t numBytes,
						   cl_bool block,
						   EventArgs args = {}) 
	{
		throwOnCLError(clEnqueueReadBuffer(
			id,
			buf.id,
			block, 					// blocking
			readOffset, 			// read offset
			numBytes,    			// num bytes to read
			dest, 					// dest ptr
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	/// Write entire buffer
	void enqueueWriteBuffer(const Buffer& dest, const void* src, cl_bool block = CL_FALSE, EventArgs args = {}) {
		enqueueWriteBuffer(dest, src, 0, dest.size, block, args);
	}
	/// Write partial buffer
	void enqueueWriteBuffer(const Buffer& dest,
							const void* src,
							size_t destOffset,
							size_t numBytes,
							cl_bool block = CL_FALSE,
							EventArgs args = {}) 
	{
		throwOnCLError(clEnqueueWriteBuffer(
			id,
			dest.id, 				// buffer
			block, 					// blocking
			destOffset, 			// write offset
			numBytes,   			// num bytes to write
			src, 					// source ptr
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueWriteBufferRect(const Buffer& dest,
								size_t destOffset,
								const void* hostPtr,
								size_t numBytes,
								cl_bool block = CL_FALSE,
								EventArgs args = {}) 
	{
		size_t buffer_origin[3] = {destOffset, 0, 0};
		size_t host_origin[3]   = {0, 0, 0};
		size_t region[3]		= {numBytes, 1, 1};
		throwOnCLError(clEnqueueWriteBufferRect(
			id,
			dest.id,                // dest buffer
			block,					// blocking
			buffer_origin,
			host_origin,
			region,
			0,						// buffer row pitch
			0,						// buffer slice pitch
			0,						// host row pitch
			0,						// host slice pitch
			hostPtr,
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueBarrier(EventArgs args = {}) {
		throwOnCLError(clEnqueueBarrierWithWaitList(
			id,
			args.numWaitEvents(),		// wait list size
			args.waitList.data(),		// event wait list
			args.event					// event
		));
	}
	/// Fill entire buffer with value.
	template<typename T>
	void enqueueFillBuffer(const Buffer& buffer, T value, EventArgs args = {}) {
		throwOnCLError(clEnqueueFillBuffer(
			id,
			buffer.id,
			&value,
			sizeof(T),				// pattern size
			0,						// offset
			buffer.size,			// size
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	/// Copy whole buffer (assumes same size)
	void enqueueCopyBuffer(const Buffer& src, const Buffer& dest, EventArgs args = {}) {
		assert(src.size==dest.size);
		enqueueCopyBuffer(src, 0ULL, dest, 0ULL, src.size, args);
	}
	/// Copy partial buffer
	void enqueueCopyBuffer(const Buffer& src, size_t srcOffset,
						   const Buffer& dest, size_t destOffset,
						   size_t numBytes,
						   EventArgs args = {}) 
	{
		throwOnCLError(clEnqueueCopyBuffer(
			id,
			src.id,
			dest.id,
			srcOffset,              // src offset
			destOffset,             // dest offset
			numBytes, 	            // num bytes
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueCopyBufferToImage(const Buffer& src,
								  const Image& dest,
								  uint width,
								  uint height,
								  EventArgs args = {}) 
	{
		ulong dest_origin[3] = {0,0,0};
		ulong region[3]	     = {width,height,1};
		throwOnCLError(clEnqueueCopyBufferToImage(
			id,
			src.id,
			dest.id,
			0,                      // src offset
			dest_origin,            // dest origin
			region,                 // region
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueWriteImage(const Image& image,
						   const void* hostPtr,
						   bool block = CL_FALSE,
						   EventArgs args = {}) 
	{
		size_t origin[3] = {0, 0, 0};
		size_t region[3] = {image.width, image.height, 1};
		throwOnCLError(clEnqueueWriteImage(
			id,
			image.id,
			block,
			origin,
			region,
			0,
			0,
			hostPtr,
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueReadImage(const Image& image,
						  void* hostPtr,
						  bool block = CL_FALSE,
						  EventArgs args = {}) 
	{
		size_t origin[3] = {0,0,0};
		size_t region[3] = {image.width,image.height,1};
		throwOnCLError(clEnqueueReadImage(
			id,
			image.id,
			block,
			origin,
			region,
			0,						// row pitch
			0,						// slice pitch
			hostPtr,
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueAcquireGLObjects(const vector<MemObject> objects, EventArgs args = {}) {
		vector<cl_mem> ids;
		for(auto& it : objects) {
			ids.push_back(it.id);
		}
		throwOnCLError(clEnqueueAcquireGLObjects(
			id,
			(uint)objects.size(),
			ids.data(),
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueReleaseGLObjects(const vector<MemObject> objects, EventArgs args = {}) {
		vector<cl_mem> ids;
		for(auto& it : objects) {
			ids.push_back(it.id);
		}
		throwOnCLError(clEnqueueReleaseGLObjects(
			id,
			(uint)objects.size(),
			ids.data(),
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	/// Maps a region of buffer into the host address space
	/// and returns a pointer to this mapped region
	void* enqueueMapBuffer(const Buffer& buf,
						   size_t offset,
						   size_t numBytes,
						   cl_map_flags flags,
						   cl_bool block = CL_FALSE,
						   EventArgs args = {}) 
	{
		assert((flags & ~(CL_MAP_READ | CL_MAP_WRITE)) == 0);
		int err;
		void* ptr = clEnqueueMapBuffer(
			id,
			buf.id,
			block,
			flags,
			offset,
			numBytes,
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event,				// event
			&err
		);
		throwOnCLError(err);
		return ptr;
	}
	/// Maps a region of image into the host address space
	/// and returns a pointer to this mapped region. rowPitch and slicePitch are also set.
	/// [[UNTESTED]]
	void* enqueueMapImage(const Image& img,
						  cl_map_flags flags,
						  ulong* rowPitch, 
						  ulong* slicePitch,
						  cl_bool block = CL_FALSE,
						  EventArgs args = {})
	{
		assert((flags & ~(CL_MAP_READ | CL_MAP_WRITE)) == 0);
		ulong origin[3] = {0,0,0};
		ulong region[3] = {img.width, img.height, 1};
		int err;
		void* ptr = clEnqueueMapImage(
			id,
			img.id,
			block,
			flags,
			origin,
			region,
			rowPitch,
			slicePitch,
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event,				// event
			&err
		);
		throwOnCLError(err);
		return ptr;
	}
	/// Unmap a previously mapped region of a buffer or image 
	void enqueueUnmapMemObject(const MemObject& buf, void* ptr, EventArgs args = {}) {
		throwOnCLError(clEnqueueUnmapMemObject(
			id,
			buf.id,
			ptr,
			args.numWaitEvents(),	// wait list size
			args.waitList.data(),	// event wait list
			args.event				// event
		));
	}
	void enqueueKernel(const Kernel& kernel,
					   vector<ulong> globalSizes,
					   vector<ulong> localSizes = {}, 
					   EventArgs args = {}) 
	{
		assert(globalSizes.size() <= 3);
		assert(localSizes.size() == 0 || localSizes.size() == globalSizes.size());
		const ulong* local = localSizes.size() == 0 ? nullptr : localSizes.data();

		int err = 0;
		err = clEnqueueNDRangeKernel(
			id, 
			kernel.id,
			(uint)globalSizes.size(),	// dimensions
			nullptr,					// always set this to null
			globalSizes.data(),			// global work sizes
			local,						// local work sizes
			args.numWaitEvents(),		// wait list size
			args.waitList.data(),		// event wait list
			args.event					// event
		);
		throwOnCLError(err);
	}
	void flush() {
		throwOnCLError(clFlush(id));
	}
	void finish() {
		throwOnCLError(clFinish(id));
	}
};

} /// opencl