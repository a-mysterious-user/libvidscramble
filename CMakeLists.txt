cmake_minimum_required(VERSION 3.22)
project(libvidscramble)

if(NOT WIN32)
    message(WARNING "This CMake configuration has only been tested on Windows")
endif()

set(CMAKE_CXX_STANDARD 17)

# specify the architecture
set(CMAKE_SYSTEM_PROCESSOR AMD64)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

OPTION(LIBVIDSCRAMBLE_BUILD_TEST "Build libvscramble tests" OFF)

find_package(OpenCV REQUIRED)

if(LIBVIDSCRAMBLE_BUILD_TEST)
    find_package(Python3 COMPONENTS NumPy REQUIRED)
endif()


add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/fmt)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/json)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/zlib)

# copy generated zconf.h
add_custom_target(
        zconf
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/third_party/zlib/zconf.h
        ${PROJECT_SOURCE_DIR}/include/zconf.h
        DEPENDS zlibstatic
)

message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
message(STATUS "Python3 headers: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 numpy headers: ${Python3_NumPy_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/third_party/fmt/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/json/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/pybind11_opencv_numpy)
include_directories(${PROJECT_SOURCE_DIR}/third_party/zlib)
include_directories(${PROJECT_SOURCE_DIR}/third_party/zstr/src)
include_directories(${PROJECT_SOURCE_DIR}/third_party/schifra)
include_directories(${PROJECT_SOURCE_DIR}/third_party/argparse/include)
include_directories(${PROJECT_SOURCE_DIR}/include)


add_library(vidscramble SHARED
        ${PROJECT_SOURCE_DIR}/include/scrambler.h
        ${PROJECT_SOURCE_DIR}/include/pipeline.h
        ${PROJECT_SOURCE_DIR}/include/pipeline_parser.h
        ${PROJECT_SOURCE_DIR}/include/scrambler_constructor_spec.h
        ${PROJECT_SOURCE_DIR}/include/util.h
        ${PROJECT_SOURCE_DIR}/include/data_embed.h
        ${PROJECT_SOURCE_DIR}/src/scrambler.cpp
        ${PROJECT_SOURCE_DIR}/src/pipeline.cpp
        ${PROJECT_SOURCE_DIR}/src/pipeline_parser.cpp
        ${PROJECT_SOURCE_DIR}/src/util.cpp
        ${PROJECT_SOURCE_DIR}/src/data_embed.cpp
        )

add_dependencies(vidscramble zconf)
target_link_libraries(vidscramble ${OpenCV_LIBRARIES} fmt::fmt zlibstatic)

if(MSVC)
    set_target_properties(vidscramble PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_link_libraries(vidscramble ws2_32)
endif()

add_executable(video_decoder ${PROJECT_SOURCE_DIR}/src/video_decoder.cpp)
target_link_libraries(video_decoder vidscramble)

pybind11_add_module(py_vidscramble MODULE
        ${PROJECT_SOURCE_DIR}/third_party/pybind11_opencv_numpy/ndarray_converter.h
        ${PROJECT_SOURCE_DIR}/third_party/pybind11_opencv_numpy/ndarray_converter.cpp
        ${PROJECT_SOURCE_DIR}/src/python_module.cpp)
target_link_libraries(py_vidscramble PRIVATE vidscramble)
target_include_directories(py_vidscramble PRIVATE ${PROJECT_SOURCE_DIR}/lib)

add_executable(test ${PROJECT_SOURCE_DIR}/test/test.cpp)
target_link_libraries(test vidscramble)

# copy dynamic libraries on windows
if(WIN32)
    get_filename_component(OpenCV_RUNTIME_DIR "${OpenCV_LIB_PATH}/../bin" ABSOLUTE)
    file(GLOB OpenCV_RUNTIME_LIBS ${OpenCV_RUNTIME_DIR}/*.dll)
    foreach(DLL_PATH ${OpenCV_RUNTIME_LIBS})
        get_filename_component(DLL_NAME ${DLL_PATH} NAME)
        add_custom_command(
                TARGET vidscramble POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy
                ${DLL_PATH}
                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DLL_NAME})
    endforeach()
endif()