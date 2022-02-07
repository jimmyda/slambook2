cmake_minimum_required(VERSION 3.11)

project(
    useEigen)

# Add header file
include_directories("/usr/include/eigen3")

ADD_EXECUTABLE( app.out eigenMatrix.cpp )