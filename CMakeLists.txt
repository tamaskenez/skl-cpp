cmake_minimum_required(VERSION 3.0)
project(sklcpp)

if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED 1)
endif()

add_subdirectory(sklearn)
