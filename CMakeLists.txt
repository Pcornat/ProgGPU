cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(ProgGPU CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)

include_directories(${OpenCV_INCLUDE_DIRS})

if (NOT ${UNIX})
	message(FATAL_ERROR "Not on an Unix-like system (Linux for example)")
endif ()

set(LINKER_FLAGS PRIVATE
	-Wl,--sort-common,--as-needed
	-flto
	-fuse-ld=gold
	-fuse-linker-plugin)

set(NOTCUDA_FILES
	omp_seq_code/mainOMP.c
	omp_seq_code/run_functions.c
	omp_seq_code/run_functions.h
	omp_seq_code/compute_functions.c
	omp_seq_code/compute_functions.h
	omp_seq_code/images.h
	omp_seq_code/images.c)

#set_source_files_properties(${NOTCUDA_FILES} PROPERTIES LANGUAGE CXX)

set(CUDA_FILES
	cuda_code/cuda_functions.cu
	cuda_code/cuda_functions.cuh
	cuda_code/mainCUDA.cu
	cuda_code/images.cu
	cuda_code/images.cuh
	cuda_code/compute_functions.cu
	cuda_code/compute_functions.cuh
	)

if (NOT ${CMAKE_HOST_UNIX})
	message(FATAL_ERROR "Program only for Linux systems.")
endif ()

#if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
#	set(NOTCUDA_OPTIONS
#		PRIVATE
#		-pipe
#		-march=native
#		-m64
#		-xHost
#		-Wall
#		-Wextra
#		${OpenMP_C_FLAGS})
#
#	set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
#	set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -Ofast")
#
#	set(CUDA_OPTIONS
#		PRIVATE
#		-m64
#		-Xcompiler -pipe,-march=native,-Wall,-Wextra)
#elseif (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
set(NOTCUDA_OPTIONS
	-pipe
	-march=native
	-m64
	-Wall
	-Wextra
	-funroll-loops
	-flto
	-fuse-linker=gold
	-fuse-linker-plugins)

set(CUDA_OPTIONS
	-m64
	-Xcompiler=-pipe,-march=native,-m64,-Wall,-Wextra,-funroll-loops,-flto,-fuse-ld=gold,-fuse-linker-plugin
	-gencode arch=compute_61,code=sm_61)
#endif ()

add_executable(ProgGPU ${CUDA_FILES})
set_target_properties(ProgGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ProgGPU BEFORE PRIVATE /opt/cuda/include)
target_compile_definitions(ProgGPU PRIVATE HAVE_CUB)
target_compile_options(ProgGPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
target_link_options(ProgGPU ${LINKER_FLAGS})
target_link_libraries(ProgGPU ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)

#add_executable(Prog ${NOTCUDA_FILES})
##set_target_properties(Prog PROPERTIES LINKER_LANGUAGE C)
#target_compile_options(Prog ${NOTCUDA_OPTIONS})
#target_link_options(Prog ${LINKER_FLAGS})
#target_link_libraries(Prog OpenMP::OpenMP_CXX ${OpenCV_LIBRARIES} m)