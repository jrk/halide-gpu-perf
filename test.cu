#include <stdint.h>
#include <stdio.h>
#include <assert.h>

const int kernel_radius = 3;
const int kernel_area = (kernel_radius*2+1)*(kernel_radius*2+1);

template <typename T>
struct buf {

    T *host;
    T *dev;
    int32_t base[2];
    int32_t extent[2];
    int32_t stride[2];
    int32_t size;

    buf(int width, int height) {
        base[0] = base[1] = 0;
        stride[0] = 1;
        stride[1] = width;
        extent[0] = width;
        extent[1] = height;
        
        size = width*height;
        host = new T[size];
		fprintf(stderr, "Allocated (%dx%d)->host at 0x%lx\n", width, height, (size_t)host);

        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&dev, size*sizeof(T));
		
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device buffer (error code %s)!\n",
							cudaGetErrorString(err));
            exit(-1);
        }
    }

    buf() {}

    void free() {
        delete host;
        cudaFree(dev);
    }

    __host__ __device__ inline T& operator()(int x, int y) {
        assert(x >= base[0] && x < extent[0] && y >= base[1] && y < extent[1]);
#if defined(__CUDA_ARCH__)
        return dev[(x-base[0])*stride[0] + (y-base[1])*stride[1]];
#else
        return host[(x-base[0])*stride[0] + (y-base[1])*stride[1]];
#endif
    }
    void h_to_d() {
        cudaMemcpy((void*)dev, (void*)host, size*sizeof(T), cudaMemcpyHostToDevice);
    }
    void d_to_h() {
        cudaMemcpy((void*)host, (void*)dev, size*sizeof(T), cudaMemcpyDeviceToHost);
    }

    template<typename Fn>
    void for_each(Fn f) {
        for (int y = base[1]; y < base[1]+extent[1]; y++) {
            for (int x = base[0]; x < base[0]+extent[0]; x++) {
                f(x, y);
            }
        }
    }
};

__global__ void
boxBlur(buf<float> in, buf<float> out) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x + out.base[0];
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y + out.base[1];

    if (x >= out.extent[0] || y >= out.extent[1]) return;
    
    float res = 0;
    for (int j = -kernel_radius; j <= kernel_radius; j++) {
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            // if (x == 100 && y == 100) {
            //     printf("res += in(%d,%d) = %.2f\n", x+i, y+j, in(x+i, y+j));
            // }
            res += in(x+i, y+j);
        }
    }
    res /= float(kernel_area);

    out(x,y) = res;

    if (x > 8 && y > 8 && x < 16 && y < 16) {
        printf("(%d,%d) %.2f => %.2f\n", x, y, in(x,y), out(x,y));
    }
}

int main (int argc, char const *argv[])
{
    const int width = 6400,
              height = 4800;
    buf<float> in(width+2*kernel_radius, height+2*kernel_radius),
               out(width, height);

    in.base[0] = in.base[1] = -kernel_radius;

    const int block_width = 16,
              block_height = 16;
    dim3 blocks((width + block_width - 1) / block_width,
                (height + block_height - 1) / block_height);
    dim3 threads(block_width, block_height);

    in.for_each([&](int x, int y) {
        in(x, y) = (x % 3 == 0 && y % 3 == 0) ? 1.f : 0.f;
    });
    in.h_to_d();

    boxBlur<<<blocks, threads>>>(in, out);

    out.d_to_h();
    in.for_each([&](int x, int y) {
        if (y > 16) return;
        if (x >= 0 && x < 16) printf("%.2f ", in(x, y));
        if (x == 16) printf("\n");
    });
	printf("\n");
    out.for_each([&](int x, int y) {
        if (y > 16) return;
        if (x >= 0 && x < 16) printf("%.2f ", out(x, y));
        if (x == 16) printf("\n");
    });
	
	in.free();
	out.free();

    return 0;
}
