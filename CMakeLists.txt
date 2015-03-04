cmake_minimum_required(VERSION 2.8.4)
project(LinearLearner)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    blas/blas.h
    blas/blasp.h
    blas/daxpy.c
    blas/ddot.c
    blas/dnrm2.c
    blas/dscal.c
    linear.cpp
    linear.h
    predict.c
    train.c
    tron.cpp
    tron.h)

add_executable(LinearLearner ${SOURCE_FILES})