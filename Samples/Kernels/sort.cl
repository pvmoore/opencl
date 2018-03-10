/**
 *  Passed in by the application:
 *
 *  WORK_GROUP_SIZE = 256
 *  ASCENDING       = true|false
 *
 */

#if ASCENDING==true
    #define COMPARE_RT(a,b) (a >= b)
    #define COMPARE_LT(a,b) (a > b)
#else
    #define COMPARE_RT(a,b) (a <= b)
    #define COMPARE_LT(a,b) (a < b)
#endif

int findOnRight(global const float* ptr,
                       const float v,
                       const uint chunkSize)
{
    int left  = 0;
    int right = chunkSize;
    int mid;

    int it = 0;
    do{
        mid = (left+right)/2;
        if(mid==left || mid==right) break;

        float value = ptr[mid];

        bool choice = COMPARE_RT(value, v);

        right = choice ? mid : right;   // go left
        left  = choice ? left : mid;    // go right

    }while(true/* && it++<1000*/);

    mid += !COMPARE_RT(ptr[mid], v);

    return mid;
}
int findOnLeft(global const float* ptr,
                      const float v,
                      const uint chunkSize)
{
    int left  = 0;
    int right = chunkSize;
    int mid;

    int it = 0;
    do{
        mid = (left+right)/2;
        if(mid==left || mid==right) break;

        float value = ptr[mid];

        bool choice = COMPARE_LT(value, v);

        right = choice ? mid : right;   // go left
        left  = choice ? left : mid;    // go right

    }while(true/* && it++<1000*/);

    mid += !COMPARE_LT(ptr[mid], v);

    return mid;
}
//================================================================
// Take 2 sorted chunks and merge them together to create a
// larger sorted chunk double the size. Call this as many times
// as required to sort the whole data set.
// 1st iteration: chunkSize will be 256 -> output 512 chunksize
//================================================================
__attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
kernel void merge(
    global const float* in,
    global float* out,
    const uint chunkSize)
{
    const int tid        = get_global_id(0);
    const int chunkSize2 = chunkSize*2;
    const int chunk      = tid/chunkSize2;
    const bool onLeft    = tid%chunkSize2 < chunkSize;
    const float v        = in[tid];

    const global float* a   = in+chunkSize2*chunk;
    const global float* b   = a+chunkSize;

    if(onLeft) {
        int pos = findOnRight(b, v, chunkSize);
        out[tid+pos] = v;
    } else {
        int pos = findOnLeft(a, v, chunkSize);
        out[(tid-chunkSize)+pos] = v;
    }
}

//================================================================
// Sorts locally only.
//================================================================
/**
 *  Sorts input into length/WORK_GROUP_SIZE sorted arrays
 *  of WORK_GROUP_SIZE floats each.
 *  Both arrays must be a power of 2 in length.
 */
 __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
kernel void bitonicSortLocal(
    global float* data)
{
    local float aux[WORK_GROUP_SIZE];
    const int i = get_local_id(0); // index in workgroup

    /// Move to block start
    const int offset = get_group_id(0) * WORK_GROUP_SIZE;
    data  += offset;

    /// Cache the local block
    aux[i] = data[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    /// Loop on sorted sequence length
    //__attribute__((opencl_unroll_hint))
#pragma unroll
    for(int length=1; length<WORK_GROUP_SIZE; length<<=1) {
        // 0=asc, 1=desc
        #if ASCENDING==true
        bool dir = ((i & (length<<1)) != 0);
        #else
        bool dir = ((i & (length<<1)) == 0);
        #endif
        /// Loop on comparison distance (between keys)
        for(int inc=length; inc>0; inc>>=1) {
            /// sibling to compare
            int j      = i ^ inc;
            float iKey = aux[i];
            float jKey = aux[j];
            bool smaller = (jKey < iKey) || (jKey == iKey && j < i);
            bool swap = smaller ^ (j < i) ^ dir;

            barrier(CLK_LOCAL_MEM_FENCE);
            aux[i] = (swap)?jKey:iKey;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    /// Write output
    data[i] = aux[i];
}

