cmake_minimum_required(VERSION 3.22)

project(chora CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

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


#
# RYOANJI
#

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(RYOANJI_INCLUDE ${PROJECT_SOURCE_DIR}/ryoanji/src)
include_directories(${RYOANJI_INCLUDE} ${MPI_INCLUDE_PATH})

add_library(ryoanji
	${RYOANJI_INCLUDE}/ryoanji/interface/treebuilder.cu
	${RYOANJI_INCLUDE}/ryoanji/interface/ewald.cu
	${RYOANJI_INCLUDE}/ryoanji/interface/multipole_holder.cu
)

#
# RYOANJI MPI DEMO
#

#set(RYOANJI_TEST ${PROJECT_SOURCE_DIR}/ryoanji/test)
#set(CSTONE_TEST ${PROJECT_SOURCE_DIR}/cstone/test)
set(CHORA_DIR ${PROJECT_SOURCE_DIR}/chora)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(chora-test
	${CHORA_DIR}/chora-test.cpp
	${CHORA_DIR}/Constants.cpp
	${CHORA_DIR}/ErfInv.cpp
	${CHORA_DIR}/Maxwellian.cpp
	${CHORA_DIR}/Random.cpp
	${CHORA_DIR}/ParticleAdvancer.cu
	${CHORA_DIR}/ParticleList.cu
	${CHORA_DIR}/util.cu
)
set_target_properties(chora-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(chora-test PUBLIC ryoanji cstone MPI::MPI_CXX)

