# C++20 order book core: header-only library, native tests, latency harness, pybind11 module.
#
#   cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/native --parallel
#   ctest --test-dir build/native --output-on-failure
#
# The Python module (_native) is written into src/exchange_engine/ so the installed package
# finds it; pass -DEE_BUILD_PYTHON=OFF to build only the C++ targets.

cmake_minimum_required(VERSION 3.20)
project(ee_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EE_BUILD_PYTHON "Build the pybind11 module exchange_engine._native" ON)
set(EE_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/exchange_engine" CACHE PATH
    "Where the _native module is written")

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

add_executable(bench_native bench/bench_native.cpp)
target_link_libraries(bench_native PRIVATE ee_core)

if(EE_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT pybind11_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _pyb_rc)
    if(NOT _pyb_rc EQUAL 0)
      message(FATAL_ERROR "pybind11 not found for ${Python_EXECUTABLE}: pip install pybind11 (or -DEE_BUILD_PYTHON=OFF)")
    endif()
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native python/binding.cpp)
  target_link_libraries(_native PRIVATE ee_core)
  # $<1:...> keeps multi-config generators (Visual Studio) from adding a per-config subdirectory.
  set_target_properties(_native PROPERTIES LIBRARY_OUTPUT_DIRECTORY "$<1:${EE_PYTHON_PACKAGE_DIR}>")
endif()
