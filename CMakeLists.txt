# CMake最低版本要求
cmake_minimum_required(VERSION 3.15)

# 项目名称和C++标准
project(LV_UNet_Deploy_CPP CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 寻找OpenCV库 ---
# 在安装了libopencv-dev的Ubuntu容器内，这行可以很好地工作
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- 寻找ONNX Runtime库 (修正版) ---
# 这个部分现在会使用从Docker构建命令中传入的 ONNXRUNTIME_DIR 变量。
# 它会检查这个变量是否被定义，如果没有（例如在你的Windows本地开发时），则使用一个默认路径。
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "E:/libs/onnxruntime-win-x64-gpu-1.18.0")
endif()

message(STATUS "Using ONNX Runtime from: ${ONNXRUNTIME_DIR}")

# 添加头文件目录，这样编译器才能找到 onnxruntime_cxx_api.h
include_directories(${ONNXRUNTIME_DIR}/include)
# 添加库文件目录，这样链接器才能找到 .lib/.so 文件
link_directories(${ONNXRUNTIME_DIR}/lib)

# --- 创建可执行文件 ---
add_executable(run_inference src/main.cpp)

# --- 链接所有需要的库 ---
target_link_libraries(run_inference
    # 链接OpenCV的核心库
    ${OpenCV_LIBS}
    # 链接ONNX Runtime的核心库
    onnxruntime
    onnxruntime_providers_shared
)