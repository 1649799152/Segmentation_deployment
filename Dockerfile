# --- 阶段一：构建环境 (Builder Stage) - 最终版 ---
# 基础镜像使用11.8.0，以匹配ONNX Runtime v1.18.0的需求
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 更换为阿里云镜像源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list

# 安装编译所需的所有依赖，包括与CUDA 11.8匹配的工具包
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    cuda-toolkit-11-8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./onnxruntime-linux-x64-gpu-1.18.0 /usr/local/onnxruntime
COPY . .
RUN cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
          -D ONNXRUNTIME_DIR=/usr/local/onnxruntime \
    && cmake --build build --config Release -j$(nproc)


# --- 阶段二：运行环境 (Runner Stage) - 最终毕业版 ---
# 基础镜像同样使用11.8.0的运行时版
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app
# 设置环境变量，让系统能在/app下寻找我们手动拷贝的ONNX Runtime库
ENV LD_LIBRARY_PATH=/app

# 同样先换源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list

# 使用apt-get安装所有运行时依赖，它会自动处理所有深层依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-dnn4.5d \
    libcudnn8 \
    libcublas-12-1 \
    && rm -rf /var/lib/apt/lists/*

# 只从构建环境中拷贝我们自己编译的程序和ONNX Runtime库
COPY --from=builder /app/build/run_inference .
COPY --from=builder /usr/local/onnxruntime/lib/libonnxruntime.so* .
COPY --from=builder /usr/local/onnxruntime/lib/libonnxruntime_providers_*.so* .

COPY ./models ./models
ENTRYPOINT ["./run_inference"]