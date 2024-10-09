cmake_minimum_required(VERSION 3.22)

project(chora CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
#set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

#
# CHORA
#

#set(RYOANJI_TEST ${PROJECT_SOURCE_DIR}/ryoanji/test)
#set(CSTONE_TEST ${PROJECT_SOURCE_DIR}/cstone/test)
set(CHORA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP) # openmp for direct CPU solve (zimmemi1)

set(CSTONE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../cstone/include)
set(RYOANJI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../ryoanji/src)

include_directories(${CSTONE_INCLUDE} ${RYOANJI_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})

add_executable(chora-test
	${CHORA_DIR}/chora-test.cpp
	${CHORA_DIR}/Constants.cpp
	${CHORA_DIR}/ErfInv.cpp
	${CHORA_DIR}/Maxwellian.cpp
	${CHORA_DIR}/Random.cpp
	${CHORA_DIR}/ParticleAdvancer.cu
	${CHORA_DIR}/ParticleList.cu
	${CHORA_DIR}/Solvers.cpp
	${CHORA_DIR}/Loaders.cpp
	${CHORA_DIR}/util.cu
)
set_target_properties(chora-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(chora-test PUBLIC ryoanji cstone MPI::MPI_CXX OpenMP::OpenMP_CXX)

