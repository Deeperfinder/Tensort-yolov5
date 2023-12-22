cmake_minimum_required(VERSION 3.0.0)
project(YOLOV5 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

# CUDA
# 指定CUDA路径
# 查找CUDA
find_package(CUDA)

# 检查CUDA是否被找到
if(CUDA_FOUND)
    message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()
set(CUDA_PATH "/usr/local/cuda")

# 指定TensorRT的包含路径和库路径
set(TRT_INC_PATH "/usr/include/x86_64-linux-gnu")
set(TRT_LIB_PATH "/usr/lib/x86_64-linux-gnu")

# 添加包含目录
include_directories(
    ${CUDA_PATH}/include
    ${TRT_INC_PATH}
)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 检查是否找到了OpenCV
if(OpenCV_FOUND)
    # 找到了OpenCV，打印一条消息和一些变量
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    message(STATUS "OpenCV Include Directories: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
else()
    # 没有找到OpenCV，打印一条错误消息
    message(FATAL_ERROR "OpenCV not found. Please specify OpenCV directory using OpenCV_DIR variable.")
endif()

# 添加库目录
link_directories(
    ${CUDA_PATH}/lib64
    ${TRT_LIB_PATH}
)

# 指定CUDA架构和代码生成选项
set(CUDA_ARCH_FLAGS
    "-gencode=arch=compute_60,code=sm_60"
    "-gencode=arch=compute_61,code=sm_61"
    "-gencode=arch=compute_70,code=sm_70"
    "-gencode=arch=compute_75,code=sm_75"
    "-gencode=arch=compute_80,code=sm_80"
    "-gencode=arch=compute_86,code=sm_86"
    "-gencode=arch=compute_89,code=sm_89"
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# # 包含目录
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
include_directories(${CUDA_INCLUDE_DIRS})

# 设置CUDA和C++编译器标志
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14 -O3 -Xcompiler -fPIC ${CUDA_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -fPIC -use_fast_math")

set(CMAKE_BUILD_TYPE "Debug")
add_executable(YOLOV5 main.cpp)
target_link_libraries(YOLOV5
                      cudart
                      nvinfer
                      nvonnxparser
                      ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
