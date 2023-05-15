#pragma once

#include "scrambler.h"
#include "data_embed.h"
#include <memory>

using pipeline_step_t = std::shared_ptr<ScramblerBase>;

struct ImageDataTransform {
    float data_region_x = 0.0f;
    float data_region_y = 0.0f;
    float data_region_height = 0.0f;
    float data_region_width = 0.0f;
    float image_region_x = 0.0f;
    float image_region_y = 0.0f;
    float image_region_height= 0.0f;
    float image_region_width = 0.0f;
    int num_data_rows = 0;
    int num_data_cols = 0;
    int original_image_region_width = 0;
    int original_image_region_height = 0;
    int original_data_region_width = 0;
    int original_data_region_height = 0;
};



class VideoScramblePipeline{
public:
    explicit VideoScramblePipeline(std::shared_ptr<std::vector<pipeline_step_t>> steps,
                                   int data_embed_block_size,
                                   int data_embed_num_rows);

    void reset_timestamp();
    void increment_timestamp();
    void set_timestamp_increment(bool val);
    int get_data_embed_interval() const;
    void set_data_embed_interval(int interval);

    void fit(const cv::Mat &img);
    cv::Mat transform(const cv::Mat &img);
    cv::Mat inverse_transform(const cv::Mat &img, const ImageDataTransform &info);
    void sync_state(const nlohmann::json &data);
    void sync_state(const std::string &data);

    static bool get_data_extraction_transform(const cv::Mat &img, ImageDataTransform &info);
    static std::string extract_data(const cv::Mat &img, const ImageDataTransform &info);
    static cv::Mat extract_image_region(const cv::Mat &img, const ImageDataTransform &info);

    std::string to_json() const;
    cv::Mat to_json_image() const;
    cv::Mat to_json_image(const cv::Mat &img) const;
    cv::Mat to_no_data_image(const cv::Mat &img) const;
private:

    void _assert_fit() const;

    std::shared_ptr<std::vector<pipeline_step_t>> _steps;
    ScramblerState _state;
    bool _transform_increment_timestamp = true;
    bool _fit = false;

    int _data_embed_block_size = 0;
    int _data_embed_num_rows = 0;
    int _data_embed_interval = 1;

    std::unique_ptr<DataEmbed> _data_embed;
};


