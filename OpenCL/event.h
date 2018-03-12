#pragma once

namespace opencl {

struct Event final {
	cl_event id = nullptr;

	~Event() {
		if(id) releaseAll(); id = nullptr;
	}

	void await() const {
		throwOnCLError(clWaitForEvents(1, &id));
	}
	uint getReferenceCount() const {
		uint value;
		throwOnCLError(clGetEventInfo(
			id,
			CL_EVENT_REFERENCE_COUNT,
			sizeof(uint),
			&value,
			nullptr
		));
		return value;
	}
	void release() const {
		throwOnCLError(clReleaseEvent(id));
	}
	inline void retain() {
		throwOnCLError(clRetainEvent(id));
	}
	void releaseAll() const {
		int refCount = getReferenceCount();
		for(auto i = 0; i<refCount; i++) {
			throwOnCLError(clReleaseEvent(id));
		}
	}
	ulong getRunTime() const {
		ulong start, end;
		throwOnCLError(clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_START, sizeof(ulong), &start, nullptr));
		throwOnCLError(clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_END, sizeof(ulong), &end, nullptr));
		return end - start;
	}
	ulong getQueuedTime() const {
		ulong start, end;
		throwOnCLError(clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_QUEUED, sizeof(ulong), &start, nullptr));
		throwOnCLError(clGetEventProfilingInfo(id, CL_PROFILING_COMMAND_START, sizeof(ulong), &end, nullptr));
		return end - start;
	}
	/// status: CL_COMPLETE or a negative error value
	void setStatus(int status) {
		throwOnCLError(clSetUserEventStatus(id, status));
	}
	/// CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE
	/// or a negative error code
	int getStatus() {
		int value;
		int err = clGetEventInfo(
			id,
			CL_EVENT_COMMAND_EXECUTION_STATUS,
			sizeof(int),
			&value,
			nullptr
		);
		throwOnCLError(err);
		return value;
	}
};

Event createUserEvent(shared_ptr<class Context> ctx);

} /// opencl