#pragma once

namespace opencl {

inline void waitFor(cl_event event) {
	throwOnCLError(clWaitForEvents(1, &event));
}
inline uint getReferenceCount(cl_event e) {
	uint value;
	throwOnCLError(clGetEventInfo(
		e,
		CL_EVENT_REFERENCE_COUNT,
		sizeof(uint),
		&value,
		nullptr
	));
	return value;
}
inline void release(cl_event e) {
	throwOnCLError(clReleaseEvent(e));
}
inline void releaseAll(cl_event e) {
	int refCount = getReferenceCount(e);
	for(auto i = 0; i<refCount; i++) {
		throwOnCLError(clReleaseEvent(e));
	}
}
inline void retain(cl_event e) {
	throwOnCLError(clRetainEvent(e));
}
inline ulong getRunTime(cl_event event) {
	ulong start, end;
	throwOnCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(ulong), &start, nullptr));
	throwOnCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(ulong), &end, nullptr));
	return end - start;
}
inline ulong getQueuedTime(cl_event event) {
	ulong start, end;
	throwOnCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(ulong), &start, nullptr));
	throwOnCLError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(ulong), &end, nullptr));
	return end - start;
}
inline cl_event createUserEvent(shared_ptr<Context> ctx) {
	int err;
	cl_event evt = clCreateUserEvent(ctx->context, &err);
	throwOnCLError(err);
	return evt;
}
/// status: CL_COMPLETE or a negative error value
inline void setStatus(cl_event e, int status) {
	throwOnCLError(clSetUserEventStatus(e, status));
}
/// CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE
/// or a negative error code
inline int getStatus(cl_event e) {
	int value;
	int err = clGetEventInfo(
		e,
		CL_EVENT_COMMAND_EXECUTION_STATUS,
		sizeof(int),
		&value,
		nullptr
	);
	throwOnCLError(err);
	return value;
}

}