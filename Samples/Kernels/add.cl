
//__attribute__((reqd_work_group_size(X, Y, Z)))
kernel void Add(global const uint* a, 
				global const uint* b, 
				global uint* c,
				const uint delta)
{
    int tid = get_global_id(0);
   
    c[tid] = a[tid] + b[tid] + delta;
}