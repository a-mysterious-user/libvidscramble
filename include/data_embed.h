#pragma once

#include "scrambler.h"
#include <zstr.hpp>
#include <array>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include "winsock.h"
#else
#include <arpa/inet.h>
#endif



/* Reed Solomon Code Parameters */
constexpr const size_t rs_code_length = 15;
constexpr const size_t rs_fec_length  = 3;
constexpr const size_t rs_data_length = rs_code_length - rs_fec_length;

/* Finite Field Parameters */
constexpr const size_t field_descriptor                =   4;
constexpr const size_t generator_polynomial_index      = 0;
constexpr const size_t generator_polynomial_root_count =  rs_fec_length;

constexpr const int data_embed_expansion = 4;

const int cv_aruco_marker_dict = cv::aruco::DICT_6X6_50;
const std::array<int, 3> cv_aruco_marker_inds{0,1,2};


std::string encode_metadata(std::initializer_list<uint16_t> data);

template<typename T>
std::vector<uint16_t> decode_metadata(const T* data, int size) {
    if(size % 2 != 0) {
        throw std::runtime_error{"invalid size"};
    }
    std::vector<uint16_t> ret(size/2, 0x0000);
    uint16_t cur_data;
    auto cur_data_p = reinterpret_cast<int8_t*>(&cur_data);
    for(auto i = 0; i < size; i+=2) {
        cur_data_p[0] = data[i] & 0xFF;
        cur_data_p[1] = data[i+1] & 0xFF;
        ret[i/2] = ntohs(cur_data);
    }
    return ret;
}

std::array<int8_t, rs_code_length> rs_encode_block(const char *data, int size);
std::array<int8_t, rs_data_length> rs_decode_block(const char *data);



class DataEmbed {
public:
    using encoded_data_t = std::vector<uint8_t>;

    DataEmbed(int block_size, int num_rows, int image_width);

    encoded_data_t encode_data(const std::string &data) const;
    cv::Mat DataEmbed::encode_no_data(const cv::Mat &img) const;

    static std::string decode_data(const encoded_data_t &enc_data);

    cv::Mat encoded_data_as_image(const cv::Mat &img, const std::string &data) const;

    size_t get_data_region_width() const;

    size_t get_data_region_height() const;

private:
    int _block_size = 0;
    int _num_rows = 0;
    int _image_width = 0;
    int _image_width_with_marker = 0;
    int _num_bits_per_block = 0;
    int _num_blocks_per_row = 0;
    int _num_bits_per_row = 0;
    int _num_bytes_per_row = 0;
    int _num_bits_total = 0;
    int _num_bytes_total = 0;
    int _fiducial_marker_size = 0;
    int _fiducial_marker_col_2 = 0;
};

std::vector<uint16_t> rs_decode_metadata(const DataEmbed::encoded_data_t &enc_data);

// represent each byte using multiple bytes
DataEmbed::encoded_data_t expand_representation(const DataEmbed::encoded_data_t &enc_data, int expansion);

DataEmbed::encoded_data_t shrink_representation(const DataEmbed::encoded_data_t &enc_data, int expansion);