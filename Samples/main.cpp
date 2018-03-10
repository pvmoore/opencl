#include "_pch.h"

using namespace core;
using std::string;
using std::vector;
using std::shared_ptr;

#include "../OpenCL/_exports.h"
using namespace opencl;

void addExample();
void enqueueExample();
void imageReadExample();
void sortExample();

int wmain(int argc, const wchar_t* argv[]) {
#ifdef _DEBUG
	_CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_CHECK_ALWAYS_DF);
#endif

	/// Display platform info
	{
		OpenCL cl;
		printf("Found %u platforms\n", cl.numPlatforms());

		auto platform = cl.createPlatform(CL_DEVICE_TYPE_GPU);
		printf("%s\n\n", platform.toString().c_str());
	}

	addExample();
	enqueueExample();
	imageReadExample();
	sortExample();

	printf("\n\nPress ENTER");
	getchar();
	return 0;
}

