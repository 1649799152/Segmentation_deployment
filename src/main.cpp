// main.cpp (最终容器化改造版)

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

// --- 预处理与字符串转换函数 (保持不变) ---
cv::Mat preprocess(cv::Mat& image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    return cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(256, 256), cv::Scalar(), true, false, CV_32F);
}

#ifdef _WIN32
std::wstring to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}
#endif


// --- 主函数 ---
int main(int argc, char* argv[]) {
    // --- 核心改动 1：修改参数检查 ---
    // 我们现在需要4个参数：程序名、输入图片、模型路径、输出图片路径
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image> <path_to_model> <path_to_output>" << std::endl;
        return -1;
    }

    // --- 核心改动 2：接收所有参数 ---
    const char* image_path = argv[1];
    const char* model_path_char = argv[2];
    const char* output_path = argv[3]; // 新增的输出路径参数
    
    #ifdef _WIN32
        std::wstring model_path_wstr = to_wstring(model_path_char);
        const wchar_t* model_path = model_path_wstr.c_str();
    #else
        const char* model_path = model_path_char;
    #endif

    try {
        // --- 1. 初始化阶段 (无改动) ---
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LV-UNet-Deploy");
        Ort::SessionOptions session_options;
        session_options.AppendExecutionProvider_CUDA({});
        Ort::Session session(env, model_path, session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "--- Model and session initialized successfully ---" << std::endl;

        // --- 2. 模型预热 (Warm-up) 阶段 (无改动) ---
        std::cout << "\n--- Warming up the model on GPU... ---" << std::endl;
        auto warmup_start = std::chrono::high_resolution_clock::now();
        cv::Mat dummy_image = cv::Mat::zeros(256, 256, CV_8UC3);
        cv::Mat dummy_blob = preprocess(dummy_image);
        std::vector<int64_t> dummy_input_shape = {1, 3, 256, 256};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value dummy_input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)dummy_blob.data, dummy_blob.total(), dummy_input_shape.data(), dummy_input_shape.size());
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        session.Run(Ort::RunOptions{nullptr}, input_names, &dummy_input_tensor, 1, output_names, 1);
        auto warmup_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> warmup_elapsed = warmup_end - warmup_start;
        std::cout << "--- Model is hot! Warm-up took: " << warmup_elapsed.count() << " ms ---" << std::endl;

        // --- 3. 真正处理用户数据阶段 (无改动) ---
        std::cout << "\n--- Processing user image: " << image_path << " ---" << std::endl;
        cv::Mat user_image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (user_image.empty()) {
            std::cerr << "Could not read the user image!" << std::endl;
            return -1;
        }
        cv::Mat user_input_blob = preprocess(user_image);
        Ort::Value user_input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)user_input_blob.data, user_input_blob.total(), dummy_input_shape.data(), dummy_input_shape.size());
        auto inference_start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &user_input_tensor, 1, output_names, 1);
        auto inference_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_elapsed = inference_end - inference_start;
        std::cout << "Actual inference time (hot): " << inference_elapsed.count() << " ms" << std::endl;

        // --- 4. 后处理并保存结果 ---
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        cv::Mat mask(256, 256, CV_32FC1, floatarr);
        cv::Mat binary_mask;
        cv::threshold(mask, binary_mask, 0.5, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);
        
        // --- 核心改动 3：使用参数指定的路径保存图片 ---
        cv::imwrite(output_path, binary_mask);
        std::cout << "Mask saved to " << output_path << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}