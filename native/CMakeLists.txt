# C++20 order book core: header-only library and its native tests.
#
#   cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/native --parallel
#   ctest --test-dir build/native --output-on-failure

cmake_minimum_required(VERSION 3.20)
project(ee_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(ee_core INTERFACE)
target_include_directories(ee_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(ee_core INTERFACE cxx_std_20)
if(MSVC)
  target_compile_options(ee_core INTERFACE /W4 /permissive-)
else()
  target_compile_options(ee_core INTERFACE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

add_executable(test_order_book tests/test_order_book.cpp)
target_link_libraries(test_order_book PRIVATE ee_core)
add_test(NAME test_order_book COMMAND test_order_book)

add_executable(test_no_alloc tests/test_no_alloc.cpp)
target_link_libraries(test_no_alloc PRIVATE ee_core)
add_test(NAME test_no_alloc COMMAND test_no_alloc)
