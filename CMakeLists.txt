cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(ProgGPU C CUDA)

set(CMAKE_C_STANDARD 90)

include_directories(.)

find_package(OpenMP REQUIRED)

#[[
if (NOT DEFINED ${UNIX})
	message(FATAL_ERROR "Not on a Unix-like system (Linux for example)")
endif ()
]]

set(LINKER_FLAGS
		-Wl,--sort-common,--as-needed)

set(OPENMP_FILES
		openmp_functions.c openmp_functions.h)

set(CUDA_FILES
		cuda_functions.cu cuda_functions.h)

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel")
	set(COMPILE_OPTIONS
			PRIVATE -pipe
			PRIVATE -march=native
			PRIVATE -Wall
			PRIVATE -Wextra
			PRIVATE -m64
			PRIVATE -xHost)

	set(LINKER_FLAGS
			${LINKER_FLAGS}
			-shared-intel)

elseif (${CMAKE_C_COMPILER_ID} STREQUAL "GNU")
	set(COMPILE_OPTIONS
			PRIVATE -pipe
			PRIVATE -march=native
			PRIVATE -Wall
			PRIVATE -Wextra)
	#[[set(LINKER_FLAGS
			${LINKER_FLAGS})]]
endif ()

message(STATUS "OpenMP link : ${OpenMP_C_LIBRARIES}")

add_executable(ProgGPU main.c ${OPENMP_FILES} ${CUDA_FILES} run_funtions.c run_funtions.h)
set_target_properties(ProgGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(ProgGPU ${COMPILE_OPTIONS} ${OpenMP_C_FLAGS})
target_link_libraries(ProgGPU ${LINKER_FLAGS} ${OpenMP_C_LIBRARIES})