#ifndef __CUDA_ARCH__
// host

// #include <HalideRuntimeCuda.h>
#include <HalideBuffer.h>
using Halide::Runtime::Buffer;

// TODO: figure out CUDA context setup to get Halide CUDA Runtime to play well with libcudart/default context
template <typename T>
void dev_malloc(Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    assert(!b->dev);
    cudaMalloc((void**)(&b->dev), buf.size_in_bytes());
}

template <typename T>
void dev_free(Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    void *dev = (void*)b->dev;
    assert(dev);
    cudaFree(dev);
    b->dev = 0;
}

template <typename T>
void dev_to_host(Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    void *dev = (void*)b->dev;
    assert(dev);
    assert(b->host);
    cudaMemcpy(b->host, dev, buf.size_in_bytes(), cudaMemcpyDeviceToHost);
}

template <typename T>
void host_to_dev(Buffer<T> &buf) {
    buffer_t *b = buf.raw_buffer();
    void *dev = (void*)b->dev;
    assert(dev);
    assert(b->host);
    cudaMemcpy(dev, b->host, buf.size_in_bytes(), cudaMemcpyHostToDevice);
}

#else

// device - simple struct decl
#define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
typedef struct buffer_t {
    /** A device-handle for e.g. GPU memory used to back this buffer. */
    uint8_t* __restrict__ dev;

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */
    uint8_t* host;

    /** The size of the buffer in each dimension. */
    int32_t extent[4];

    /** Gives the spacing in memory between adjacent elements in the
    * given dimension.  The correct memory address for a load from
    * this buffer at position x, y, z, w is:
    * host + elem_size * ((x - min[0]) * stride[0] +
    *                     (y - min[1]) * stride[1] +
    *                     (z - min[2]) * stride[2] +
    *                     (w - min[3]) * stride[3])
    * By manipulating the strides and extents you can lazily crop,
    * transpose, and even flip buffers without modifying the data.
    */
    int32_t stride[4];

    /** Buffers often represent evaluation of a Func over some
    * domain. The min field encodes the top left corner of the
    * domain. */
    int32_t min[4];

    /** How many bytes does each buffer element take. This may be
    * replaced with a more general type code in the future. */
    int32_t elem_size;

    /** This should be true if there is an existing device allocation
    * mirroring this buffer, and the data has been modified on the
    * host side. */
    HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;

    /** This should be true if there is an existing device allocation
    mirroring this buffer, and the data has been modified on the
    device side. */
    HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;

    // Some compilers will add extra padding at the end to ensure
    // the size is a multiple of 8; we'll do that explicitly so that
    // there is no ambiguity.
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
} buffer_t;

#endif
