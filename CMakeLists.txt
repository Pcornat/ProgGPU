cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ProgGPU C CUDA)

set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)
find_package(OpenCV 3.4.5 REQUIRED)

include_directories(.)
include_directories(${OpenCV_INCLUDE_DIRS})

if (NOT ${UNIX})
	message(FATAL_ERROR "Not on an Unix-like system (Linux for example)")
endif ()

set(LINKER_FLAGS
		-Wl,--sort-common,--as-needed)

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
		cuda_code/useless.c)

if (NOT ${CMAKE_HOST_UNIX})
	message(FATAL_ERROR "Program only for Linux systems.")
endif ()

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
	set(NOTCUDA_OPTIONS
			PRIVATE
			-pipe
			-march=native
			-m64
			-xHost
			-Wall
			-Wextra
			${OpenMP_C_FLAGS})

	set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
	set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -Ofast")

	set(CUDA_OPTIONS
			PRIVATE
			-m64
			-Xcompiler -pipe,-march=native,-Wall,-Wextra)
elseif (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
	set(NOTCUDA_OPTIONS
			PRIVATE
			-pipe
			-march=native
			-Wall
			-Wextra
			${OpenMP_C_FLAGS})
endif ()

add_executable(ProgGPU ${CUDA_FILES})
set_target_properties(ProgGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ProgGPU BEFORE PRIVATE /opt/cuda/include)
target_compile_options(ProgGPU ${CUDA_OPTIONS})
target_link_libraries(ProgGPU ${LINKER_FLAGS} -shared-libgcc ${OpenCV_LIBRARIES})

add_executable(Prog ${NOTCUDA_FILES})
#set_target_properties(Prog PROPERTIES LINKER_LANGUAGE C)
target_compile_options(Prog ${NOTCUDA_OPTIONS})
target_link_libraries(Prog ${LINKER_FLAGS} ${OpenMP_C_LIBRARIES} -cxxlib m ${OpenCV_LIBRARIES})