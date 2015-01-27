cmake_minimum_required(VERSION 2.8.4)
project(MLx)

find_package(Boost COMPONENTS system filesystem REQUIRED)
set(BLAS_LIB openblas)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse4.1")
include_directories(${Boost_INCLUDE_DIRS} blas/include)
link_directories(blas/lib)

file(GLOB_RECURSE SRC
    src/*.h
    src/*.c
    src/*.cpp
)

add_executable(ML ${SRC})
target_link_libraries(ML ${Boost_LIBRARIES} ${BLAS_LIB})