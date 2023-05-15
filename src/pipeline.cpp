#include "pipeline.h"

#include <iostream>

VideoScramblePipeline::VideoScramblePipeline(std::shared_ptr<std::vector<pipeline_step_t>> steps,
                                             int data_embed_block_size,
                                             int data_embed_num_rows) : _steps(steps),
                                                                        _state(),
                                                                        _data_embed_block_size(data_embed_block_size),
                                                                        _data_embed_num_rows(data_embed_num_rows),
                                                                        _transform_increment_timestamp(true)
                                                                        {

}

void VideoScramblePipeline::reset_timestamp() {
    _state.timestamp = 0;
}

void VideoScramblePipeline::increment_timestamp() {
    ++_state.timestamp;
}

void VideoScramblePipeline::set_timestamp_increment(bool val) {
    _transform_increment_timestamp = val;
}

void VideoScramblePipeline::fit(const cv::Mat &img) {
    if (img.type() != CV_8UC3) {
        throw std::runtime_error{"only supports 3 channel ubyte image"};
    }

    _state.input_height = img.rows;
    _state.input_width = img.cols;

    cv::Mat cur_img(img);
    for(const pipeline_step_t &step : *_steps){
        step->fit(_state, cur_img);
        cur_img = step->transform(_state, cur_img);
    }

    _state.timestamp = 0;
    _state.output_width_wo_data = cur_img.cols;
    _state.output_height_wo_data = cur_img.rows;

    _data_embed = std::make_unique<DataEmbed>(_data_embed_block_size, _data_embed_num_rows, _state.output_width_wo_data);

    _state.data_region_height = _data_embed->get_data_region_height();
    _state.data_region_width = _data_embed->get_data_region_width();

    _fit = true;
}

cv::Mat VideoScramblePipeline::transform(const cv::Mat &img) {
    if (img.type() != CV_8UC3) {
        throw std::runtime_error{"only supports 3 channel ubyte image"};
    }

    _assert_fit();
    cv::Mat cur_img(img);

    for(const pipeline_step_t &step : *_steps){
        cur_img = step->transform(_state, cur_img);
    }

    cv::Mat ret;
    if(_state.timestamp % _data_embed_interval == 0) {
        ret = to_json_image(cur_img);
    } else {
        ret = to_no_data_image(cur_img);
    }

    if(_transform_increment_timestamp){
        increment_timestamp();
    }

    return ret;
}

cv::Mat VideoScramblePipeline::inverse_transform(const cv::Mat &img, const ImageDataTransform &info) {
    _assert_fit();

    if (img.type() != CV_8UC3) {
        throw std::runtime_error{"only supports 3 channel ubyte image"};
    }

    cv::Mat cur_img(img);

    // extract image region
    cur_img = extract_image_region(img, info);

    for(auto iter = _steps->rbegin(); iter != _steps->rend(); ++iter){
        cur_img = (*iter)->inverse_transform(_state, cur_img);
    }

    if(_transform_increment_timestamp){
        increment_timestamp();
    }

    return cur_img;
}


void VideoScramblePipeline::_assert_fit() const {
    if(!_fit){
        throw std::runtime_error{format("[VideoScramblePipeline] the fit() function must be called before use")};
    }
}

std::string VideoScramblePipeline::to_json() const {
    _assert_fit();
    nlohmann::ordered_json ret;
    std::vector<nlohmann::json> steps;
    for(const pipeline_step_t &step : *_steps){
        steps.emplace_back(step->to_json());
    }
    ret["steps"] = steps;

    ret["data_embed_block_size"] = _data_embed_block_size;
    ret["data_embed_num_rows"] =_data_embed_num_rows;
    ret["data_embed_interval"] = _data_embed_interval;
//    ret["rs_code_length"] = rs_code_length;
//    ret["rs_fec_length"] = rs_fec_length;
//    ret["field_descriptor"] = field_descriptor;
//    ret["generator_polynomial_index"] = generator_polynomial_index;
//    ret["generator_polynomial_root_count"] = generator_polynomial_root_count;

    nlohmann::ordered_json state;
    state["output_width_wo_data"] = _state.output_width_wo_data;
    state["output_height_wo_data"] = _state.output_height_wo_data;
    state["data_region_width"] = _state.data_region_width;
    state["data_region_height"] = _state.data_region_height;
    state["input_height"] = _state.input_height;
    state["input_width"] = _state.input_width;
    state["timestamp"] = _state.timestamp;


    ret["state"] = state;

    return ret.dump();
}

cv::Mat VideoScramblePipeline::to_json_image(const cv::Mat &img) const {
    auto json_s = to_json();
    return _data_embed->encoded_data_as_image(img, json_s);
}

cv::Mat VideoScramblePipeline::to_json_image() const {
    cv::Mat dummy_img(_state.output_height_wo_data, _state.data_region_width, CV_8UC3);
    return to_json_image(dummy_img);
}

cv::Mat VideoScramblePipeline::to_no_data_image(const cv::Mat &img) const {
    return _data_embed->encode_no_data(img);
}



bool VideoScramblePipeline::get_data_extraction_transform(const cv::Mat &img, ImageDataTransform &info) {
    if (img.type() != CV_8UC3) {
        throw std::runtime_error{"only supports 3 channel ubyte image"};
    }

    auto aruco_dict = cv::aruco::getPredefinedDictionary(cv_aruco_marker_dict);
    cv::Mat marker;
    cv::aruco::generateImageMarker(aruco_dict, cv_aruco_marker_inds[0], 32, marker);

    std::vector<int> marker_inds;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
    cv::aruco::ArucoDetector detector(aruco_dict, detector_params);
    detector.detectMarkers(img, marker_corners, marker_inds);

    // try to find markers
    auto marker_0_find = std::find(marker_inds.begin(), marker_inds.end(), cv_aruco_marker_inds[0]);
    if (marker_0_find == marker_inds.end()) {
        std::cerr << "unable to find the bottom left fiducial marker\n";
        return false;
    }

    auto marker_1_find = std::find(marker_inds.begin(), marker_inds.end(), cv_aruco_marker_inds[1]);
    if (marker_1_find == marker_inds.end()) {
        std::cerr << "unable to find the bottom right fiducial marker\n";
        return false;
    }

    auto marker_2_find = std::find(marker_inds.begin(), marker_inds.end(), cv_aruco_marker_inds[2]);
    if (marker_1_find == marker_inds.end()) {
        std::cerr << "unable to find the top right fiducial marker\n";
        return false;
    }

    auto marker_0_ind = std::distance(marker_inds.begin(), marker_0_find);
    auto marker_1_ind = std::distance(marker_inds.begin(), marker_1_find );
    auto marker_2_ind = std::distance(marker_inds.begin(), marker_2_find );

    // scale correction based on the bottom left marker
    const auto &marker_0_corners = marker_corners[marker_0_ind];
    std::vector<float> dim_vals;
    dim_vals.reserve(marker_0_corners.size());
    for(auto p : marker_0_corners) {
        dim_vals.push_back(p.x);
    }

    float x_min_0 = *std::min_element(dim_vals.begin(), dim_vals.end()),
        x_max_0 = *std::max_element(dim_vals.begin(), dim_vals.end());

    dim_vals.clear();
    for(auto p : marker_0_corners) {
        dim_vals.push_back(p.y);
    }

    float y_min_0 = *std::min_element(dim_vals.begin(), dim_vals.end()),
        y_max_0 = *std::max_element(dim_vals.begin(), dim_vals.end());


    // get rough estimate of block sizes
    float x_span = x_max_0 - x_min_0, y_span = y_max_0 - y_min_0;

    float block_size_x = x_span / 4.0f;
    float block_size_y = y_span / 4.0f;
    if(block_size_x < 2.0f || block_size_y < 2.0f) {
        std::cerr << format("detected block size is ({},{}), which is too small\n", block_size_x, block_size_y);
        return false;
    }

    // compute location for the bottom right marker
    dim_vals.clear();
    const auto &marker_1_corners = marker_corners[marker_1_ind];
    for(auto p : marker_1_corners) {
        dim_vals.push_back(p.x);
    }
    float x_min_1 = *std::min_element(dim_vals.begin(), dim_vals.end()),
            x_max_1 = *std::max_element(dim_vals.begin(), dim_vals.end());
    dim_vals.clear();
    for(auto p : marker_1_corners) {
        dim_vals.push_back(p.y);
    }
    float y_min_1 = *std::min_element(dim_vals.begin(), dim_vals.end()),
            y_max_1 = *std::max_element(dim_vals.begin(), dim_vals.end());


    // compute location for the top right marker
    dim_vals.clear();
    const auto &marker_2_corners = marker_corners[marker_2_ind];
    for(auto p : marker_2_corners) {
        dim_vals.push_back(p.x);
    }
    float x_min_2 = *std::min_element(dim_vals.begin(), dim_vals.end()),
            x_max_2 = *std::max_element(dim_vals.begin(), dim_vals.end());
    dim_vals.clear();
    for(auto p : marker_2_corners) {
        dim_vals.push_back(p.y);
    }
    float y_min_2 = *std::min_element(dim_vals.begin(), dim_vals.end()),
            y_max_2 = *std::max_element(dim_vals.begin(), dim_vals.end());

    // extract data in the first block using the rough estimates
    float dr_x_0 = x_max_0 + block_size_x;
    float dr_y_0 = y_min_0 + block_size_y / 2;

    auto num_metadata_rs_code = (6 / rs_data_length) + (6 % rs_data_length != 0); // metadata field has length 6
    auto num_metadata_rs_block = num_metadata_rs_code * (rs_code_length / 3) * data_embed_expansion;
    std::vector<uint8_t> code_buf(num_metadata_rs_block * 3, 0x00);

    bool decode_success = false;
    // the estimate of block_size_x is unreliable
    // we need to use brute force to find a good value
    for(auto delta_t = 0; delta_t <= 10; ++delta_t) {

        if(decode_success){
            break;
        }

        for(auto sign : {1.0f, -1.0f}) {
            if(sign < 0.0f && delta_t == 0){
                continue;
            }

            float block_size_x_change_factor = 1.0 + delta_t * sign * 0.01;
            float block_size_x_changed = block_size_x * block_size_x_change_factor;

            for(auto i = 0; i < num_metadata_rs_code; ++i) {
                for(auto j = 0; j < num_metadata_rs_block; ++j) {
                    float pix_x = dr_x_0 + j * block_size_x_changed;
                    const cv::Vec3b &pix_val = img.at<cv::Vec3b>(lround(dr_y_0), lround(pix_x));
                    code_buf[3 * j] = pix_val[0];
                    code_buf[3 * j + 1] = pix_val[1];
                    code_buf[3 * j + 2] = pix_val[2];
                }
            }

            // try decoding this block
            std::vector<uint8_t> shrunk_code_buf;
            try{
                shrunk_code_buf = shrink_representation(code_buf, data_embed_expansion);
            } catch (const std::exception &e) {
                std::cerr << format("[delta={}] unable to shrink binary representation: {}\n", block_size_x_change_factor, e.what());
                continue;
            }

            std::vector<uint16_t> metadata;
            try {
                metadata = rs_decode_metadata(shrunk_code_buf);
            } catch (const std::exception &e) {
                std::cerr << format("[delta={}] unable to decode metadata information: {}\n", block_size_x_change_factor, e.what());
                continue;
            }

            constexpr const int max_num_row = 24;
            constexpr const int max_num_col = 3840 / 4;

            if (metadata[0] <= 0) {
                std::cerr << format("[delta={}] invalid number of rows ({}) detected\n", block_size_x_change_factor, metadata[0]);
                continue;
            }


            if (metadata[1] <= 0) {
                std::cerr << format("[delta={}] invalid number of rows ({}) detected\n", block_size_x_change_factor, metadata[0]);
                continue;
            }

            if(metadata[0] > max_num_row) {
                std::cerr << format("[delta={}] parsed number of rows ({}) is greater than max allowed number of rows ({})\n", block_size_x_change_factor, metadata[0], max_num_row);
                continue;
            }
            if(metadata[1] > max_num_col) {
                std::cerr << format("[delta={}]  parsed number of cols ({}) is greater than max allowed number of cols ({})\n", block_size_x_change_factor, metadata[1], max_num_col);
                continue;
            }


            info.data_region_x = x_max_0 + block_size_x / 2;
            info.data_region_y = y_min_0;
            info.data_region_height = y_max_1 - y_min_0;
            info.data_region_width = (x_min_1 - block_size_x / 2) - (x_max_0 + block_size_x / 2);
            info.num_data_rows = metadata[0];
            info.num_data_cols = metadata[1];
            info.image_region_x = x_min_0 - block_size_x / 2;
            info.image_region_y = y_min_2;
            info.image_region_width = x_min_2 - block_size_x/2 - info.image_region_x;
            info.image_region_height = y_min_1 - block_size_y / 2 - info.image_region_y;

            std::string decoded_data;
            try {
                decoded_data = extract_data(img, info);
            } catch (const std::exception &e) {
                std::cerr << format("[delta={}] an error occurred trying to parse data: {}\n", block_size_x_change_factor, e.what());
                continue;
            }

            nlohmann::json parsed_data;
            try {
                parsed_data = nlohmann::json::parse(decoded_data);
                info.original_image_region_width = parsed_data["state"]["output_width_wo_data"].get<int>();
                info.original_image_region_height = parsed_data["state"]["output_height_wo_data"].get<int>();
                info.original_data_region_width = parsed_data["state"]["data_region_width"].get<int>();
                info.original_data_region_height = parsed_data["state"]["data_region_height"].get<int>();
            } catch (const std::exception &e) {
                std::cerr << format("[delta={}] an error occurred trying to parse data as JSON: {}\n", block_size_x_change_factor, e.what());
                continue;
            }



            decode_success = true;
            break;
        }
    }

    if(!decode_success) {
        std::cerr << "failed to find image transform parameters after searching for block_size_x\n";
        return false;
    }

    return true;
}

std::string VideoScramblePipeline::extract_data(const cv::Mat &img, const ImageDataTransform &info) {

    float block_size_x = info.data_region_width / info.num_data_cols;
    float block_size_y = info.data_region_height / info.num_data_rows;
    float start_x = info.data_region_x + block_size_x / 2;
    float start_y = info.data_region_y + block_size_y / 2;

    cv::Mat new_img;
    img.copyTo(new_img);

    DataEmbed::encoded_data_t encoded_data(3 * info.num_data_cols * info.num_data_rows, 0x00);

    for(auto i = 0; i < info.num_data_rows; ++i) {
        float y = start_y + i * block_size_y;
        auto row_offset = i * info.num_data_cols;
        for(auto j = 0; j < info.num_data_cols; ++j) {
            float x = start_x + j * block_size_x;
            const auto &pix = img.at<cv::Vec3b>(lround(y), lround(x));
            encoded_data[3 * (row_offset + j)] = pix[0];
            encoded_data[3 * (row_offset + j) + 1] = pix[1];
            encoded_data[3 * (row_offset + j) + 2] = pix[2];
        }
    }


    return DataEmbed::decode_data(encoded_data);
}


cv::Mat get_padded_roi(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height) {
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    cv::Mat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0) {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0) {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols) {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows) {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        cv::Rect R(top_left_x, top_left_y, width, height);
        copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_REFLECT);
    }
    else {
        // no border padding required
        cv::Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
}

cv::Mat VideoScramblePipeline::extract_image_region(const cv::Mat &img, const ImageDataTransform &info) {
    // estimate transformation scale

    auto roi = get_padded_roi(img, lround(info.image_region_x), lround(info.image_region_y),
                                lround(info.image_region_width), lround(info.image_region_height));

    cv::resize(roi, roi, cv::Size(info.original_image_region_width, info.original_image_region_height));

    return roi;
}

void VideoScramblePipeline::sync_state(const nlohmann::json &data) {
    _state.timestamp = data["timestamp"].get<size_t>();
}

void VideoScramblePipeline::sync_state(const std::string &data) {
    auto json_data = nlohmann::json::parse(data);
    sync_state(json_data);
}

int VideoScramblePipeline::get_data_embed_interval() const {
    return _data_embed_interval;
}

void VideoScramblePipeline::set_data_embed_interval(int interval) {
    if(interval < 1) {
        throw std::runtime_error{"data embed interval must be at least 1"};
    }
    _data_embed_interval = interval;
}


