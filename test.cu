#include <stdint.h>
#include <stdio.h>
#include <assert.h>
// #include <HalideRuntimeCuda.h>
#include <HalideBuffer.h>

const int kernel_radius = 9;
const int kernel_area = (kernel_radius*2+1)*(kernel_radius*2+1);
const int width = 6400,
          height = 4800;

#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

template <typename T>
ALWAYS_INLINE inline
__device__ T& dev_pixel(buffer_t &buf, int x, int y) {
    T *data = (T*)buf.dev;
    assert(buf.stride[0] == 1);
    assert(buf.stride[1] > 6300 && buf.stride[1] < 6500);
    int x_offset = (x - buf.min[0]);
    int y_offset = (y - buf.min[1]) * buf.stride[1];
    return data[x_offset + y_offset];
}

__global__ void
boxBlur(buffer_t in, buffer_t out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + out.min[0];
    int y = blockIdx.y * blockDim.y + threadIdx.y + out.min[1];

    if (x < out.min[0] || y < out.min[1] ||
        x >= out.extent[0] || y >= out.extent[1])
    {
        return;
    }
    
    float res = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++) {
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            res += dev_pixel<float>(in, x+i, y+j);
        }
    }
    res *= 1.f/kernel_area;

    dev_pixel<float>(out, x,y) = res;
}

// TODO: restrict on in/out DOUBLES performance!
// #define _FLOAT(b) (reinterpret_cast<float*>((b).dev))
#define OUT_PIXEL(x,y) (out[(x)+6400*(y)])
#define IN_PIXEL(x,y) (in[((x)+kernel_radius)+6400*((y)+kernel_radius)])
__global__ void
boxBlurStatic(const float * __restrict__ in, float * __restrict__ out) {
// boxBlurStatic(const float *in, float *out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float res = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++) {
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            res += IN_PIXEL(x+i, y+j);
        }
    }
    res /= float(kernel_area);

    OUT_PIXEL(x,y) = res;
}
// #undef _FLOAT
#undef OUT_PIXEL
#undef IN_PIXEL

template <typename T>
void dev_malloc(Halide::Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    assert(!b->dev);
    cudaMalloc((void**)(&b->dev), buf.size_in_bytes());
}

template <typename T>
void dev_free(Halide::Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    void *dev = (void*)b->dev;
    assert(dev);
    cudaFree(dev);
    b->dev = 0;
}

template <typename T>
void dev_to_host(Halide::Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    void *dev = (void*)b->dev;
    assert(dev);
    assert(b->host);
    cudaMemcpy(b->host, dev, buf.size_in_bytes(), cudaMemcpyDeviceToHost);
}

template <typename T>
void host_to_dev(Halide::Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    void *dev = (void*)b->dev;
    assert(dev);
    assert(b->host);
    cudaMemcpy(dev, b->host, buf.size_in_bytes(), cudaMemcpyHostToDevice);
}

int main (int argc, char const *argv[])
{
    int trials = 1;
    if (argc == 2) {
        trials = atoi(argv[1]);
    }
    Halide::Buffer<float> in(width+2*kernel_radius, height+2*kernel_radius),
               out(width, height);

    in.set_min(-kernel_radius, -kernel_radius);

    const int block_width = 32,
              block_height = 32;
    dim3 blocks((width + block_width - 1) / block_width,
                (height + block_height - 1) / block_height);
    dim3 threads(block_width, block_height);

    in.for_each_element([&](int x, int y) {
        in(x, y) = (x % 3 == 0 && y % 3 == 0) ? 1.f : 0.f;
    });
    
    dev_malloc(in);
    dev_malloc(out);
    
    host_to_dev(in);
    host_to_dev(out);

    cudaEvent_t startEv, endEv;
    cudaEventCreate(&startEv);
    cudaEventCreate(&endEv);

    cudaEventRecord(startEv);
    for (int i = 0; i < trials; i++) {
        // boxBlur<<<blocks, threads>>>(*in.raw_buffer(), *out.raw_buffer());
        boxBlurStatic<<<blocks, threads>>>((float*)in.raw_buffer()->dev, (float*)out.raw_buffer()->dev);
    }
    cudaEventRecord(endEv);

    dev_to_host(in);
    dev_to_host(out);

    in.for_each_element([&](int x, int y) {
        if (y > 16) return;
        if (x >= 0 && x < 16) printf("%.2f ", in(x, y));
        if (x == 16) printf("\n");
    });
    printf("\n");
    out.for_each_element([&](int x, int y) {
        if (y > 16) return;
        if (x >= 0 && x < 16) printf("%.2f ", out(x, y));
        if (x == 16) printf("\n");
    });

    float elapsed;
    cudaEventElapsedTime(&elapsed, startEv, endEv);
    printf("\n-------\nTIME: %f ms / %d trials = %f ms\n", elapsed, trials, elapsed/trials);
    int64_t pixels = width*height;
    int64_t kernel_pixels = (kernel_radius*2+1)*(kernel_radius*2+1);
    printf("Inputs accumulated: %ldM\n", pixels*kernel_pixels/1000000);
    
    dev_free(in);
    dev_free(out);
    
    return 0;
}
