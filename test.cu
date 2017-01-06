#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "test_buffer.h"

const int kernel_radius = 9;
const int kernel_area = (kernel_radius*2+1)*(kernel_radius*2+1);
const int width = 6400,
          height = 4800;

#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

template <typename T, bool static_addr=false>
ALWAYS_INLINE inline
__device__ T& write_pixel(buffer_t buf, int x, int y) {
    T *data = (T*)buf.dev;
    int x_min, y_min, x_stride, y_stride;
    if (static_addr) {
        assert(buf.stride[0] == 1);
        assert(buf.stride[1] > 6300 && buf.stride[1] < 6500);
        x_min = buf.min[0];
        y_min = buf.min[1];
        x_stride = 1;
        y_stride = buf.stride[1];
    } else {
        x_min = 0;
        y_min = 0;
        x_stride = 1;
        y_stride = width;
    }
    int x_offset = (x - x_min) * x_stride;
    int y_offset = (y - y_min) * y_stride;
    return data[x_offset + y_offset];
}

template <typename T, bool static_addr=false>
ALWAYS_INLINE inline
const __device__ T read_pixel(const buffer_t buf, int x, int y) {
    const T *data = (const T*)buf.dev;
    int x_min, y_min, x_stride, y_stride;
    if (static_addr) {
        assert(buf.stride[0] == 1);
        assert(buf.stride[1] > 6300 && buf.stride[1] < 6500);
        x_min = buf.min[0];
        y_min = buf.min[1];
        x_stride = 1;
        y_stride = buf.stride[1];
    } else {
        x_min = -kernel_radius;
        y_min = -kernel_radius;
        x_stride = 1;
        y_stride = width+2*kernel_radius;
    }
    int x_offset = (x - x_min) * x_stride;
    int y_offset = (y - y_min) * y_stride;
    return __ldg(data + x_offset + y_offset);
}

__global__ void
boxBlurBuf(const buffer_t in, buffer_t out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + out.min[0];
    int y = blockIdx.y * blockDim.y + threadIdx.y + out.min[1];

#if 0
    if (x < out.min[0] || y < out.min[1] ||
        x >= out.extent[0] || y >= out.extent[1])
    {
        return;
    }
#endif
    
    float res = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++) {
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            res += read_pixel<float>(in, x+i, y+j);
        }
    }
    res /= float(kernel_area);

    write_pixel<float>(out, x,y) = res;
}

__global__ void
boxBlurBufStatic(const buffer_t in, buffer_t out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + out.min[0];
    int y = blockIdx.y * blockDim.y + threadIdx.y + out.min[1];

    float res = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++) {
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            res += read_pixel<float,true>(in, x+i, y+j);
        }
    }
    res /= kernel_area;

    write_pixel<float,true>(out, x,y) = res;
}

// TODO: restrict on in/out DOUBLES performance!
#define OUT_PIXEL(x,y) (out[(x)+width*(y)])
#define IN_PIXEL(x,y) (in[((x)+kernel_radius)+width*((y)+kernel_radius)])
__global__ void
boxBlurStatic(const float * __restrict__ in, float *out) {
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

__global__ void
boxBlurStaticNonRestrict(const float *in, float *out) {
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
#undef OUT_PIXEL
#undef IN_PIXEL

#ifndef __CUDA_ARCH__

const int block_width = 32,
          block_height = 32;
dim3 blocks((width + block_width - 1) / block_width,
            (height + block_height - 1) / block_height);
dim3 threads(block_width, block_height);

using std::vector;
using std::string;
using std::pair;

// template<class ...Args>
// void variant(std::string name, void(*kernel)(Args...), Args... args) {
//     variants.push_back(
//         std::make_pair(name, [&]{
//             kernel<<<blocks, threads>>>(args...);
//         })
//     );
// }
#define variant(nm,...) (variants.push_back(std::make_pair( \
#nm, [&]{\
    (nm)<<<blocks,threads>>>(__VA_ARGS__); \
} \
)))

int main (int argc, char const *argv[])
{
    int trials = 1;
    if (argc == 2) {
        trials = atoi(argv[1]);
    }
    Buffer<float> in(width+2*kernel_radius, height+2*kernel_radius),
               out(width, height);

    in.set_min(-kernel_radius, -kernel_radius);

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
    
    typedef std::function<void(void)> Fn;
    vector<pair<string,Fn> > variants;
    
    variant(boxBlurBuf,
            *(in.raw_buffer()), *(out.raw_buffer()) );
    
    variant(boxBlurBufStatic,
            *(in.raw_buffer()), *(out.raw_buffer()) );
    
    variant(boxBlurStatic,
            (float*)in.raw_buffer()->dev, (float*)out.raw_buffer()->dev);
    
    variant(boxBlurStaticNonRestrict,
            (float*)in.raw_buffer()->dev, (float*)out.raw_buffer()->dev);

    for (auto &variant : variants)
    {
        std::string name;
        Fn fn;
        std::tie(name, fn) = variant;
        
        cudaEventRecord(startEv);
        for (int i = 0; i < trials; i++) {
            fn();
        }
        cudaEventRecord(endEv);

        dev_to_host(in);
        dev_to_host(out);

        float elapsed;
        cudaEventElapsedTime(&elapsed, startEv, endEv);
        printf( "\n-------\n"
                "%s\n"
                "TIME: %f ms / %d trials = %f ms\n",
                name.c_str(), elapsed, trials, elapsed/trials );
        int64_t pixels = width*height;
        int64_t kernel_pixels = (kernel_radius*2+1)*(kernel_radius*2+1);
        printf("Inputs accumulated: %ldM\n", pixels*kernel_pixels/1000000);
    }

    dev_free(in);
    dev_free(out);
    
    return 0;
}
#endif //host-only