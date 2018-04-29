cmake_minimum_required(VERSION 3.9)
project(CVC)

set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR /home/vadim/project/opencv-3.3.1/cmake-build-debug)

set(OpenCV_FOUND 1)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)

add_executable(CVC ${SOURCE_FILES})
target_link_libraries(CVC ${OpenCV_LIBS})