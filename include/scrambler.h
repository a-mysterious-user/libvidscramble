#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <any>
#include <unordered_map>
#include <string>
#include <vector>
#include <random>
#include "util.h"
#include <nlohmann/json.hpp>


using random_geneator_t = std::mt19937;

struct ScramblerState{
    size_t timestamp = 0;
    size_t output_width_wo_data = 0;
    size_t output_height_wo_data = 0;
    size_t data_region_width = 0;
    size_t data_region_height = 0;
    size_t input_width = 0;
    size_t input_height = 0;
};


class ScramblerBase {
public:
    explicit ScramblerBase() : _fit(false) {}
    virtual void fit(ScramblerState &state, const cv::Mat &img) = 0;
    virtual cv::Mat transform(ScramblerState &state, const cv::Mat &img) const = 0;
    virtual cv::Mat inverse_transform(ScramblerState &state, const cv::Mat &img) const = 0;
    virtual nlohmann::json to_json() const = 0;
protected:

    void _assert_fit() const {
        if(!_fit){
            throw std::runtime_error{"the fit() function must be called before use"};
        }
    }

    bool _fit;
};


class ImageTranspose : public ScramblerBase {
public:
    void fit(ScramblerState &state, const cv::Mat &img) override;
    cv::Mat transform(ScramblerState &state, const cv::Mat &img) const override;
    cv::Mat inverse_transform(ScramblerState &state, const cv::Mat &img) const override;
    nlohmann::json to_json() const override;
};



class RowShuffle : public ScramblerBase {
public:
    explicit RowShuffle(int row_group_size, int random_seed=0);

    void fit(ScramblerState &state, const cv::Mat &img) override;
    cv::Mat transform(ScramblerState &state, const cv::Mat &img) const override;
    cv::Mat inverse_transform(ScramblerState &state, const cv::Mat &img) const override;
    nlohmann::json to_json() const override;
private:
    int _row_group_size = 0;
    int _random_seed = 0;
    int _pad = 0;
    int _num_row_groups = 0;
    int _num_rows = 0; // number of rows in the input image
    int _num_rows_after_pad = 0;
    std::vector<int> _forward_permutation;
};


class RowMix : public ScramblerBase {
public:
    explicit RowMix(int row_group_size, int random_seed);

    void fit(ScramblerState &state, const cv::Mat &img) override;
    cv::Mat transform(ScramblerState &state, const cv::Mat &img) const override;
    cv::Mat inverse_transform(ScramblerState &state, const cv::Mat &img) const override;
    nlohmann::json to_json() const override;
private:

    cv::Mat _transform_impl(ScramblerState &state, const cv::Mat &img, bool inverse) const;

    int _random_seed = 0;
    int _row_group_size = 0;
    int _num_row_groups = 0;
    int _num_rows = 0;
    std::vector<int> _forward_permutation;
};


class ImageShift : public ScramblerBase {
public:
    explicit ImageShift(int sx, int sy);

    void fit(ScramblerState &state, const cv::Mat &img) override;
    cv::Mat transform(ScramblerState &state, const cv::Mat &img) const override;
    cv::Mat inverse_transform(ScramblerState &state, const cv::Mat &img) const override;
    nlohmann::json to_json() const override;
private:
    int _sx = 0;
    int _sy = 0;
};
