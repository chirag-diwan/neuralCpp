cmake_minimum_required(VERSION 3.16)
project(NeuralNetwork VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Explicitly list your translation units (.cpp files)
# Do NOT list .h files here. They are pulled in via #include.
file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

find_package(BLAS REQUIRED)
find_package(raylib REQUIRED)

target_link_libraries(NeuralNetwork PRIVATE ${BLAS_LIBRARIES})
target_link_libraries(NeuralNetwork PRIVATE raylib)

target_compile_options(NeuralNetwork PRIVATE
)

# 2. Define exactly where the compiler should look for #include <...>
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils
)

set(XOR_TEST OFF CACHE bool "Enable XOR test")
set(MNIST_TEST OFF CACHE bool "Enable MNIST test")
set(INTERACTIVE_INFERENCE OFF CACHE bool "Enable MNIST test")

if (XOR_TEST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE XOR_TEST)
endif()

if (MNIST_TEST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MNIST_TEST)
endif()

if (INTERACTIVE_INFERENCE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE INTERACTIVE_INFERENCE)
endif()

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic )
endif()
