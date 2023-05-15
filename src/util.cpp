#include "util.h"

std::string get_opencv_mat_dt_string(OpenCVMatDT v) {
    switch (v) {
        case OpenCVMatDT::SIGNED_INT:
            return std::string{"SIGNED_INT"};
        case OpenCVMatDT::UNSIGNED_INT:
            return std::string{"UNSIGNED_INT"};
        case OpenCVMatDT::FLOAT:
            return std::string{"FLOAT"};
        default:
            return std::string{"UNKNOWN"};
    }
}

OpenCVMatDataInfo get_opencv_mat_data_info(int type) {
    uint8_t depth = type & CV_MAT_DEPTH_MASK;
    uint8_t num_channels = 1 + (type >> CV_CN_SHIFT);

    OpenCVMatDataInfo info;
    info.num_channels = num_channels;

    if (depth == CV_8U) {
        info.element_size = sizeof(uint8_t);
        info.data_type = OpenCVMatDT::UNSIGNED_INT;
    } else if(depth == CV_8S) {
        info.element_size = sizeof(int8_t);
        info.data_type = OpenCVMatDT::SIGNED_INT;
    } else if(depth == CV_16U) {
        info.element_size = sizeof(uint16_t);
        info.data_type = OpenCVMatDT::UNSIGNED_INT;
    } else if(depth == CV_16S) {
        info.element_size = sizeof(int16_t);
        info.data_type = OpenCVMatDT::SIGNED_INT;
    } else if(depth == CV_32S) {
        info.element_size = sizeof(int32_t);
        info.data_type = OpenCVMatDT::SIGNED_INT;
    } else if(depth == CV_32F) {
        info.element_size = sizeof(float);
        info.data_type = OpenCVMatDT::FLOAT;
    } else if(depth == CV_64F) {
        info.element_size = sizeof(double);
        info.data_type = OpenCVMatDT::FLOAT;
    } else {
        throw std::runtime_error{format("unknown OpenCV matrix data type \"{}\"", depth)};
    }

    return info;
}

cv::Mat translate_wrap(const cv::Mat &input, int sx, int sy) {
    // Get image dimensions.
    int w = input.size().width;
    int h = input.size().height;

    if (sx < 0) {
        sx = (-sx / w + 1) * w + sx;
    }
    if (sy < 0) {
        sy = (-sy / h + 1) * h + sy;
    }

    sx = sx % w;
    sy = sy % h;

    if (sx == 0 && sy == 0){
        return input;
    }

    // Initialize output with same dimensions and type.
    cv::Mat output = cv::Mat(h, w, input.type());

    if (sx != 0) {
        input(cv::Rect(0, sy, sx, h - sy)).copyTo(output(cv::Rect(w - sx, 0, sx, h - sy)));
    }
    if (sy != 0) {
        input(cv::Rect(sx, 0, w - sx, sy)).copyTo(output(cv::Rect(0, h - sy, w - sx, sy)));
    }
    if (sx != 0 && sy != 0) {
        input(cv::Rect(0, 0, sx, sy)).copyTo(output(cv::Rect(w - sx, h - sy, sx, sy)));
    }

    // Copy proper contents manually.
    input(cv::Rect(sx, sy, w - sx, h - sy)).copyTo(output(cv::Rect(0, 0, w - sx, h - sy)));


    return output;
}

