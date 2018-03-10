
/// pseudo random (0.0 - 0.99999)
inline float rand(float x, float y) {
	float2 co = (float2)(x, y);
	float a = 12.9898f;
	float b = 78.233f;
	float c = 43758.5453f;
	float dt = dot(co.xy, (float2)(a, b));
	float sn = fmod(dt, 3.14f);
	float s = sin(sn)*c;
	float fl;
	return fract(s, &fl);
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
					      CLK_ADDRESS_NONE | 
					      CLK_FILTER_NEAREST;

kernel void RandomImageRead(image2d_t image,
                            global uchar* output)
{
    int i    = get_global_id(0);
    int l    = get_local_id(0);
	int2 dim = get_image_dim(image);

	/// Select a random pixel to read from
	uint x     = (uint)(rand(i, l) * dim.x);
	uint y     = (uint)(rand(i, l) * dim.y);
	/// Read the pixel
	uint4 v    = read_imageui(image, sampler, (float2)(x,y));
	/// Write the result to output
	output[i]  = v.x;
}



