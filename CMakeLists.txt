cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(ProgGPU C CUDA)

set(CMAKE_C_STANDARD 99)

include_directories(. cuda_code/ omp_seq_code/)

find_package(OpenMP REQUIRED)

if (NOT ${UNIX})
	message(FATAL_ERROR "Not on a Unix-like system (Linux for example)")
endif ()

set(LINKER_FLAGS
		-Wl,--sort-common,--as-needed)

set(NOTCUDA_FILES
		omp_seq_code/mainOMP.c
		omp_seq_code/run_functions.c
		omp_seq_code/run_functions.h
		omp_seq_code/seq_functions.c
		omp_seq_code/seq_functions.h
		omp_seq_code/openmp_functions.c
		omp_seq_code/openmp_functions.h)

set(CUDA_FILES
		cuda_code/cuda_functions.cu cuda_code/cuda_functions.h cuda_code/mainCUDA.cu)

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
			-Wextra)
endif ()

message(STATUS "OpenMP link : ${OpenMP_C_LIBRARIES}")

add_executable(ProgGPU ${CUDA_FILES})
set_target_properties(ProgGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(ProgGPU ${CUDA_OPTIONS})
target_link_libraries(ProgGPU ${LINKER_FLAGS} -shared-libgcc)

add_executable(Prog ${NOTCUDA_FILES})
target_compile_options(Prog ${NOTCUDA_OPTIONS})
target_link_libraries(Prog ${LINKER_FLAGS} ${OpenMP_C_LIBRARIES} -shared-intel)