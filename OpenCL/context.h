#pragma once

namespace opencl {

class Context {
public:
	Device& device;
	cl_context context;

	Context(cl_context context, Device& device) : context(context), device(device) {}
	~Context() {  
		if(context) clReleaseContext(context);
	}

	CommandQueue createQueue(bool profiling) {
		cl_command_queue_properties command_queue_properties = 0;
		if(profiling) {
			command_queue_properties |= CL_QUEUE_PROFILING_ENABLE;
		}

		cl_queue_properties queueProperties[] = {
			CL_QUEUE_PROPERTIES,
			command_queue_properties,
			0
		};

		int err;
		auto queueId = clCreateCommandQueueWithProperties(
			context,
			device.id,
			queueProperties,
			&err
		);
		throwOnCLError(err);
		return CommandQueue{queueId};
	}
	/// Create a buffer with one or more of the following flags:
	///		CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE
	///		CL_MEM_HOST_READ_ONLY, CL_MEM_HOST_WRITE_ONLY, CL_MEM_HOST_NO_ACCESS
	///		CL_MEM_ALLOC_HOST_PTR, CL_MEM_COPY_HOST_PTR, CL_MEM_USE_HOST_PTR
	Buffer createDeviceBuffer(ulong numBytes, cl_mem_flags flags, void* hostPtr=nullptr) {
		int err;
		cl_mem bufferId = clCreateBuffer(context, flags, numBytes, hostPtr, &err);
		throwOnCLError(err);
		return Buffer{bufferId, flags, numBytes};
	}
	Image createDeviceImage(cl_mem_flags flags,
							cl_image_format format,
							cl_image_desc desc,
							void* hostPtr)
	{
		int err;
		cl_mem id = clCreateImage(
			context,
			flags,
			&format,
			&desc,
			hostPtr,
			&err
		);
		throwOnCLError(err);
		return Image{id, flags, desc.image_width, desc.image_height};
	}
	/// flags:  CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY or CL_MEM_READ_WRITE
	/// target: eg GL_TEXTURE_2D
	Buffer createFromGLTexture(cl_mem_flags flags,
							   uint textureId,
							   uint target,
							   size_t numBytes,
							   int mipLevel = 0)
	{
		int err;
		cl_mem id = clCreateFromGLTexture(
			context,
			flags,
			target,
			mipLevel,
			textureId,
			&err
		);
		throwOnCLError(err);
		return Buffer{id, flags, numBytes};
	}
	/// Create an OpenCL2.0 queue that supports device kernel_enqueue.
	CommandQueue createDeviceQueue(ulong size = 262144L) {
		auto queueProps = 0ULL |
			CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
			CL_QUEUE_ON_DEVICE |
			CL_QUEUE_ON_DEVICE_DEFAULT;

		ulong props[] = {
			CL_QUEUE_PROPERTIES,
			queueProps,
			CL_QUEUE_SIZE,
			size,
			0ULL
		};
		int err;
		auto queueId = clCreateCommandQueueWithProperties(
			context,
			device.id,
			props,
			&err
		);
		throwOnCLError(err);
		return CommandQueue{queueId};
	}
	Program createProgram(const string& filename, vector<string> options = {}) {
		return Program{context, device, filename, options};
	}
};

} /// opencl