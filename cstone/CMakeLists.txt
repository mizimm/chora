cmake_minimum_required(VERSION 3.22)

#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
#set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

#
# CSTONE
#

set(CSTONE_INCLUDE ${PROJECT_SOURCE_DIR}/cstone/include)
include_directories(${CSTONE_INCLUDE})

add_library(cstone
	${CSTONE_INCLUDE}/cstone/traversal/collisions_gpu.cu
	${CSTONE_INCLUDE}/cstone/focus/rebalance_gpu.cu
	${CSTONE_INCLUDE}/cstone/focus/source_center_gpu.cu
	${CSTONE_INCLUDE}/cstone/halos/gather_halos_gpu.cu
	${CSTONE_INCLUDE}/cstone/tree/csarray_gpu.cu
	${CSTONE_INCLUDE}/cstone/tree/octree_gpu.cu
	${CSTONE_INCLUDE}/cstone/util/reallocate.cu
	${CSTONE_INCLUDE}/cstone/primitives/primitives_gpu.cu
	${CSTONE_INCLUDE}/cstone/sfc/sfc_gpu.cu
)


