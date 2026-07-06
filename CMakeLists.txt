cmake_minimum_required(VERSION 3.20)
project(mnist_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # generates build/compile_commands.json

# Default to Debug so assertions (bounds checks, shape checks) stay live under
# ctest. Pass -DCMAKE_BUILD_TYPE=Release explicitly for optimized training runs.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra)

# Core library: the tensor + autograd.
add_library(tensor STATIC src/tensor.cpp)
target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(tensor PUBLIC cxx_std_23)

# Training executable (only if its source exists yet).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/train.cpp)
  add_executable(train apps/train.cpp)
  target_link_libraries(train PRIVATE tensor)
endif()

# Tests: any tests/*.cpp becomes a ctest case.
enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE tensor)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
