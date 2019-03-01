cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(ProgGPU C CUDA)
project(ProgOMP C)

set(CMAKE_C_STANDARD 99)

include_directories(.)

find_package(OpenMP REQUIRED)

if (NOT ${UNIX})
	message(FATAL_ERROR "Not on a Unix-like system (Linux for example)")
endif ()

set(LINKER_FLAGS
		-Wl,--sort-common,--as-needed)

set(SEQ_FILES
		seq_functions.c seq_functions.h)

set(OPENMP_FILES
		openmp_functions.c openmp_functions.h)

set(CUDA_FILES
		cuda_functions.cu cuda_functions.h)

if (NOT ${CMAKE_HOST_UNIX})
	message(FATAL_ERROR "Program only for Linux systems.")
endif ()

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
	set(INTEL_OPTIONS
			-pipe
			-march=native
			-m64
			-xHost
			-Wall
			-Wextra)

	set(COMPILE_OPTIONS
			PRIVATE $<$<COMPILE_LANGUAGE:C>:${INTEL_OPTIONS}>
			#[[$<$<COMPILE_LANGUAGE:CUDA>:-ccbin icc --compile-options="${INTEL_OPTIONS}">]])

elseif (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
	set(GNU_OPTIONS
			-pipe
			-march=native
			-Wall
			-Wextra)

	set(COMPILE_OPTIONS
			PRIVATE $<$<COMPILE_LANGUAGE:C>:${GNU_OPTIONS}>)
	#[[set(LINKER_FLAGS
			${LINKER_FLAGS})]]
endif ()

message(STATUS "OpenMP link : ${OpenMP_C_LIBRARIES}")

add_executable(ProgGPU mainCUDA.c ${CUDA_FILES} run_funtions.c run_funtions.h utils.c utils.h compute_functions.cpp compute_functions.h)
set_target_properties(ProgGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(ProgGPU ${COMPILE_OPTIONS})
target_link_libraries(ProgGPU ${LINKER_FLAGS} -shared-libgcc)

add_executable(ProgOMP mainOMP.c ${OPENMP_FILES} ${SEQ_FILES} run_funtions.c run_funtions.h utils.c utils.h compute_functions.cpp compute_functions.h)
target_compile_options(ProgOMP ${COMPILE_OPTIONS})
target_link_libraries(ProgOMP ${LINKER_FLAGS} ${OpenMP_C_LIBRARIES} ${OpenMP_C_FLAGS})