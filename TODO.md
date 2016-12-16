- [ ] Clean up - factor into a standard runnable utility class
    - Copy needed data to/from device
    - Marshal arguments
    - Variadic template for base?
    - Runner with timers
    - [Separate linking][cuda-separate-linking]
- [ ] Test with prefetch into shared (unnecessary with `__ldg`?)
- [ ] Run until statistical confidence on benchmark


# Reference
- [Straight-line scalar optimizations][straight-line-scalar-opt]
- [halide-blur performance investigation][halide-blur-perf]




[halide-blur-perf]: https://github.com/halide/Halide/issues/1568#issuecomment-266886141

[straight-line-scalar-opt]: https://docs.google.com/document/d/1momWzKFf4D6h8H3YlfgKQ3qeZy5ayvMRh6yR-Xn2hUE/view

[cuda-separate-linking]: https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code/
