#include "scrambler.h"


// trivial
void ImageTranspose::fit(ScramblerState &state, const cv::Mat &img) {_fit = true;}

cv::Mat ImageTranspose::transform(ScramblerState &state, const cv::Mat &img) const {
    cv::Mat ret;
    cv::transpose(img, ret);
    return ret;
}

cv::Mat ImageTranspose::inverse_transform(ScramblerState &state, const cv::Mat &img) const {
    cv::Mat ret;
    cv::transpose(img, ret);
    return ret;
}

nlohmann::json ImageTranspose::to_json() const {
    nlohmann::json ret;
    ret["name"] = "ImageTranspose";
    return ret;
}


RowShuffle::RowShuffle(int row_group_size, int random_seed) : _row_group_size(row_group_size),
                                                             _random_seed(random_seed)
                                                             {
    if(_row_group_size <= 0) {
        throw std::runtime_error{format("invalid line group size {} (value must be greater than zero)", _row_group_size)};
    }
}


void RowShuffle::fit(ScramblerState &state, const cv::Mat &img) {
    // determine how much to pad
    auto rows_mod = img.rows % _row_group_size;
    if(rows_mod > 0){
        _pad = _row_group_size - rows_mod;
    }

    // record number of rows
    _num_rows = img.rows;
    _num_rows_after_pad = _num_rows + _pad;
    // compute number of row groups
    _num_row_groups = _num_rows_after_pad / _row_group_size;

    // build the permutations
    random_geneator_t rand_generator(_random_seed);

    // generate the forward permutation
    _forward_permutation.resize(_num_row_groups, 0);
    for(auto i = 0; i < _forward_permutation.size(); ++i) {
        _forward_permutation[i] = i;
    }
    // shuffle
    std::shuffle(_forward_permutation.begin(), _forward_permutation.end(), rand_generator);

    _fit = true;
}

cv::Mat RowShuffle::transform(ScramblerState &state, const cv::Mat &img) const {
    _assert_fit();

    // shape check
    if(img.rows != _num_rows) {
        throw std::runtime_error{format("expected {} rows in the input image, get {}", _num_rows, img.rows)};
    }

    cv::Mat img_pad = img; // shallow copy
    // pad
    if(_pad > 0) {
        cv::copyMakeBorder(img, img_pad, 0, _pad, 0, 0, cv::BORDER_REFLECT);
    }

    // generate result
    cv::Mat ret(img_pad.rows, img_pad.cols, img_pad.type());


    // forward permutation
    for(auto i = 0; i < _forward_permutation.size(); ++i) {
        auto perm_loc = _forward_permutation[i];
        auto row_start_src = i * _row_group_size;
        auto row_start_dst = perm_loc * _row_group_size;
        for(auto j = 0; j < _row_group_size; ++j) {
            img_pad.row(row_start_src + j).copyTo(ret.row(row_start_dst + j));
        }
    }

    return ret;
}

cv::Mat RowShuffle::inverse_transform(ScramblerState &state, const cv::Mat &img) const {
    _assert_fit();

    // shape check
    if(img.rows != _num_rows_after_pad) {
        throw std::runtime_error{format("expected {} rows in the input image, get {}", _num_rows_after_pad, img.rows)};
    }

    cv::Mat ret(_num_rows, img.cols, img.type());

    // backwards permutation
    for(auto i = 0; i < _forward_permutation.size(); ++i) {
        auto perm_loc = _forward_permutation[i];
        auto row_start_src = perm_loc * _row_group_size;
        auto row_start_dst = i * _row_group_size;
        for(auto j = 0; j < _row_group_size; ++j) {
            auto dst_row = row_start_dst + j;
            if (dst_row >= ret.rows) {
                break;
            }
            img.row(row_start_src + j).copyTo(ret.row(dst_row));
        }
    }

    return ret;
}

nlohmann::json RowShuffle::to_json() const {
    nlohmann::json ret;
    ret["name"] = "RowShuffle";
    ret["row_group_size"] = _row_group_size;
    ret["random_seed"] = _random_seed;
    return ret;
}


RowMix::RowMix(int row_group_size, int random_seed) : _row_group_size(row_group_size), _random_seed(random_seed) {
    if (_row_group_size <= 0) {
        throw std::runtime_error{"in row mixing, row_group_size must be at least 1"};
    }
}

void RowMix::fit(ScramblerState &state, const cv::Mat &img) {
    // record number of rows
    _num_rows = img.rows;

    if(_num_rows % 2 != 0) {
        throw std::runtime_error{format("to enable row mixing, the number of rows must be even; "
                                 "now the number of rows is {}", _num_rows)};
    }

    if(_num_rows % _row_group_size != 0) {
        throw std::runtime_error{format("row_group_size ({}) must divide number of rows({})", _row_group_size, _num_rows)};
    }

    // build the permutations
    random_geneator_t rand_generator(_random_seed);

    auto q = _num_rows / _row_group_size;
    auto r = _num_rows % _row_group_size;
    _num_row_groups = q;
    if (r > 0) {
        ++_num_row_groups;
    }

    // generate the forward permutation
    _forward_permutation.resize(_num_row_groups, 0);
    for(auto i = 0; i < _forward_permutation.size(); ++i) {
        _forward_permutation[i] = i;
    }
    // shuffle
    std::shuffle(_forward_permutation.begin(), _forward_permutation.end(), rand_generator);

    _fit = true;
}

cv::Mat RowMix::_transform_impl(ScramblerState &state, const cv::Mat &img, bool inverse) const {
    _assert_fit();

    // shape check
    if(img.rows != _num_rows) {
        throw std::runtime_error{format("expected {} rows in the input image, get {}", _num_rows, img.rows)};
    }

    // divide rows into two groups
    auto num_rows_per_group = _num_rows / 2;

    // get the type info of the matrix
    auto mat_type_info = get_opencv_mat_data_info(img.type());

    if(mat_type_info.data_type != OpenCVMatDT::UNSIGNED_INT || mat_type_info.element_size != 1) {
        throw std::runtime_error{
                format("row mixing only supports uint8_t data type as input, get data type of {} with size {}",
                       get_opencv_mat_dt_string(mat_type_info.data_type), mat_type_info.element_size)
        };
    }

    // build a new matrix of bigger type for storage
    int new_row_dt = CV_16S;
    cv::Mat row0, row1;
    cv::Mat row_a, row_b;

    cv::Mat ret(img.rows, img.cols, img.type());

    for(auto i = 0; i < num_rows_per_group; ++i) {
        auto row_sum_group_ind = i / _row_group_size;
        auto row_sum_group_offset = i % _row_group_size;

        auto row_diff_group_ind = (i + num_rows_per_group) / _row_group_size;
        auto row_diff_group_offset = (i + num_rows_per_group) % _row_group_size;

        auto row_sum_row = _row_group_size * _forward_permutation[row_sum_group_ind] + row_sum_group_offset;
        auto row_diff_row = _row_group_size * _forward_permutation[row_diff_group_ind] + row_diff_group_offset;

        if (!inverse) {
            // forward transform

            img.row(i).convertTo(row0, new_row_dt);
            img.row(i + num_rows_per_group).convertTo(row1, new_row_dt);

            row_a = (row0 + row1) / 2;
            row_b = (row0 - row1) / 2;

            // put the sum information in the ret
            row_a.convertTo(ret.row(row_sum_row), img.type()); // we need this intermediate step because row0 and ret need to have the same type

            // need to map the negative side of row_diff to positive for uint8 encoding
            auto num_elements = row_b.total() * row_b.channels();
            auto row_diff_data = reinterpret_cast<int16_t*>(row_b.data);
            for(size_t j = 0; j < num_elements; ++j) {
                if(row_diff_data[j] < 0) {
                    row_diff_data[j] = (int16_t)256 + row_diff_data[j];
                }
            }

            row_b.convertTo(ret.row(row_diff_row), img.type());

        } else {

            img.row(row_sum_row).convertTo(row0, new_row_dt); // row sum
            img.row(row_diff_row).convertTo(row1, new_row_dt); // row diff

            // remap row_diff
            auto num_elements = row1.total() * row1.channels();
            auto row1_data = reinterpret_cast<int16_t*>(row1.data);
            for(size_t j = 0; j < num_elements; ++j) {
                if(row1_data[j] > 127) {
                    row1_data[j] = row1_data[j] - (int16_t)256;
                }
            }

            row_a = row0 + row1;
            row_b = row0 - row1;

            row_a.convertTo(ret.row(i), img.type());
            row_b.convertTo(ret.row(i+num_rows_per_group), img.type());
        }

    }


    return ret;
}


cv::Mat RowMix::transform(ScramblerState &state, const cv::Mat &img) const {
    return _transform_impl(state, img, false);
}

cv::Mat RowMix::inverse_transform(ScramblerState &state, const cv::Mat &img) const {
    return _transform_impl(state, img, true);
}

nlohmann::json RowMix::to_json() const {
    nlohmann::json ret;
    ret["name"] = "RowMix";
    ret["row_group_size"] = _row_group_size;
    ret["random_seed"] = _random_seed;
    return ret;
}

ImageShift::ImageShift(int sx, int sy) : _sx(sx), _sy(sy) {

}

void ImageShift::fit(ScramblerState &state, const cv::Mat &img) {
    _fit = true;
}

cv::Mat ImageShift::transform(ScramblerState &state, const cv::Mat &img) const {
    auto ts = state.timestamp;
    return translate_wrap(img, ts * _sx, ts * _sy);
}

cv::Mat ImageShift::inverse_transform(ScramblerState &state, const cv::Mat &img) const {
    auto ts = state.timestamp;
    return translate_wrap(img, -ts * _sx, -ts * _sy);
}

nlohmann::json ImageShift::to_json() const {
    nlohmann::json ret;
    ret["name"] = "ImageShift";
    ret["sx"] = _sx;
    ret["sy"] = _sy;
    return ret;
}
