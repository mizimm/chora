
# Chora - a GPU plasma physics solver based on Ryoanji

Ryoanji is a Barnes-Hut N-body solver for gravity and electrostatics.
It employs [EXAFMM](https://github.com/exafmm/exafmm) multipole kernels and a Barnes-Hut tree-traversal
algorithm inspired by [Bonsai](https://github.com/treecode/Bonsai). Octrees and domain decomposition are
handled by [Cornerstone Octree](https://github.com/sekelle/cornerstone-octree), see Ref. [1].

Ryoanji is optimized to run efficiently on both AMD and NVIDIA GPUs, though a CPU implementation is provided as well.

Chora reinterprets Ryoanji masses and gravitational fields as electric charges and electric fields, and provides capabilities for loading particles, advancing phase space, and reading/writing results.

## Folder structure

```
Ryoanji.git
├── README.md
├── cstone          - Cornerstone library: octree building and domain decomposition
│                     (git subtree of https://github.com/sekelle/cornerstone-octree)
│                             
└── ryoanji                            - Ryoanji: N-body solver
│                             
└── chora                            - Chora: plasma physics layer
   ├── src
   └── matlab
```



## Compilation

Chora is written in C++ and CUDA. The host `.cpp` translation units require a C++20 compiler
(GCC 11 and later, Clang 14 and later), while `.cu` translation units are compiled in the C++17 standard.
CUDA version: 11.6 or later, HIP version 5.2 or later.
CMake version: 3.25 or later.

Currently the only explicitly supported GPU is NVIDIA CUDA, A100 GPU. Run the bash script 'mkbuild' from the top-level directory, then 'cd build' and 'make'.

## Performance

## Accuracy and correctness

The entry point 'chora/chora-test.cpp' sets up a sphere of ions and electrons in thermal equilibrium. The thermal speed of the electrons is much greater than the ions because of their difference in mass. As the simulation starts, electrons will rush outward, leaving the much slower ions behind; this creates an outward electric field that pulls the electrons back, and accelerates the ions outward.

This plasma-expansion test case has three electric-field solution methods (uncomment the one you want to use and recompile):
* solveDirectCpu (OpenMP CPU brute force)
* solveDirectGpu (GPU brute force)
* solveMultipoleGpu (GPU Barnes-Hut)

The output .bin files can be read by Matlab using the 'matlab/readbin.m' script.

## References

[1] [S. Keller et al. 2023, Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations](https://doi.org/10.1145/3592979.3593417)

## Authors

* **Mike Zimmerman**
* **Sebastian Keller**
* **Rio Yokota**
