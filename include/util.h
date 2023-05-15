#pragma once

#include <opencv2/highgui.hpp>
#include <fmt/format.h>
#include <string>
#include <iostream>

using fmt::format;

enum class OpenCVMatDT {
    SIGNED_INT,
    UNSIGNED_INT,
    FLOAT
};

struct OpenCVMatDataInfo {
    OpenCVMatDT data_type;
    size_t element_size;
    size_t num_channels;
};

std::string get_opencv_mat_dt_string(OpenCVMatDT v);

OpenCVMatDataInfo get_opencv_mat_data_info(int type);

cv::Mat translate_wrap(const cv::Mat& input, int sx, int sy);