# LAMFD-UNet C++ & Docker 部署项目

这是一个综合性的AI模型部署项目，旨在完整地展示将一个基于PyTorch的深度学习模型（LAMFD-UNet），通过C++和ONNX Runtime进行高性能、跨平台的生产级部署，并最终实现容器化的全过程。

这个仓库是我的 **[《AI部署副业探索之旅》](https://blog.csdn.net/love1649799152?type=ask)** 系列技术博客的配套代码实现。欢迎关注我的探索历程，与我一同解决工程落地中的各种真实挑战。

---

## 🚀 关于模型 (About the Model)

**LAMFD-UNet** 是一个针对医学影像分割任务设计的轻量化（Lightweight）U-Net变体结构。它在保持高分割精度的同时，力求模型的计算复杂度和参数量最优化，使其更适合在对延迟和资源有严格要求的实际应用场景中进行部署。


---

## ✨ 项目亮点 (Key Features)

* **高性能C++推理**：使用纯C++实现端到端的推理流程，包括预处理和后处理，最大化地压榨硬件性能，推理速度相比Python脚本提升超过 **350倍**。
* **跨平台推理引擎**：基于 **ONNX Runtime**，利用其强大的跨平台能力和对多种硬件加速（如CUDA）的良好支持。
* **工业级容器化部署**：提供一个经过反复调试、基于**多阶段构建**的`Dockerfile`，生成一个包含所有依赖（CUDA, cuDNN, OpenCV等）的、可移植的、健壮的生产级镜像。
* **专业的调试与优化历程**：完整地复现并解决了从环境配置、编译链接、运行时依赖，到GPU冷启动、性能瓶颈分析等一系列部署中的经典难题。

---

## 🛠️ 技术栈 (Technology Stack)

* **语言**: C++ (17)
* **构建系统**: CMake
* **推理框架**: ONNX Runtime (GPU w/ CUDA)
* **图像处理**: OpenCV
* **容器化**: Docker

---

## ⚡ 快速开始 (Quick Start)

本项目设计为在Docker容器中一键构建和运行。

1.  **前置条件**:
    * 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 并配置好WSL2后端。
    * 确保你的NVIDIA驱动已更新。

2.  **构建镜像**:
    ```bash
    # 在项目根目录下运行
    docker build -t LAMFD-UNet-cpp:production .
    ```

3.  **运行推理**:
    ```bash
    # 创建一个output文件夹用于存放结果
    mkdir output

    # 运行容器，并将结果输出到output文件夹
    docker run --gpus all -v "$(pwd)/images:/app/images" -v "$(pwd)/output:/app/output" LAMFD-UNet-cpp:production /app/images/ISIC_0000001.jpg /app/models/LAMFD_UNet_simplified.onnx /app/output/mask.png
    ```
    推理结束后，分割结果`mask.png`将会出现在你本地的`output`文件夹中。

---

## 📖 探索日志

想了解这个项目从0到1的完整心路历程，以及每一个Bug的详细解决方案吗？欢迎关注我的微信公众号 **《小白部署不输》** 或者[个人CSDN博客](https://blog.csdn.net/love1649799152?type=ask)，阅读 **《AI部署副业探索之旅》** 系列文章！

![个人微信公众号](assets/qrcode_for_gh_194521ba2f3d_258.jpg)
