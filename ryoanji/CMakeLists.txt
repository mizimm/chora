cmake_minimum_required(VERSION 3.22)

project(chora CXX CUDA)
set(CMAKE_CXX_STANDARD 20)

#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
#set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CSTONE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../cstone/include)
include_directories(${CSTONE_INCLUDE})

find_package(MPI REQUIRED)

set(RYOANJI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${RYOANJI_INCLUDE} ${MPI_INCLUDE_PATH})

add_library(ryoanji
	${RYOANJI_INCLUDE}/ryoanji/interface/treebuilder.cu
	${RYOANJI_INCLUDE}/ryoanji/interface/ewald.cu
	${RYOANJI_INCLUDE}/ryoanji/interface/multipole_holder.cu
)
