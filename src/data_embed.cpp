#include "data_embed.h"
#include <cmath>

#include <schifra_galois_field.hpp>
#include <schifra_galois_field_polynomial.hpp>
#include <schifra_sequential_root_generator_polynomial_creator.hpp>
#include <schifra_reed_solomon_encoder.hpp>
#include <schifra_reed_solomon_decoder.hpp>
#include <schifra_reed_solomon_block.hpp>
#include <schifra_error_processes.hpp>
#include <schifra_reed_solomon_block.hpp>
#include <schifra_reed_solomon_bitio.hpp>



class ReedSolomon {
public:
    /* Instantiate Encoder and Decoder (Codec) */
    using rs_encoder_t = schifra::reed_solomon::encoder<rs_code_length,rs_fec_length,rs_data_length>;
    using rs_decoder_t = schifra::reed_solomon::decoder<rs_code_length,rs_fec_length,rs_data_length>;
    using rs_block_t = schifra::reed_solomon::block<rs_code_length, rs_fec_length>;

    std::unique_ptr<rs_encoder_t> _rs_encoder;
    std::unique_ptr<rs_decoder_t> _rs_decoder;
    std::unique_ptr<schifra::galois::field> _rs_field;
    std::unique_ptr<schifra::galois::field_polynomial> _rs_field_poly;
};


std::string encode_metadata(std::initializer_list<uint16_t> data) {
    std::string ret(data.size() * 2, 0x00);
    auto data_len_p = reinterpret_cast<uint16_t*>(ret.data());

    auto item_ind = 0;
    for(auto item : data) {
        *(data_len_p + item_ind) = htons(item);
        ++item_ind;
    }
    return ret;
}


ReedSolomon &get_rs_impl() {
    static std::unique_ptr<ReedSolomon> rs_impl;
    if(!rs_impl) {
        rs_impl = std::make_unique<ReedSolomon>();
        rs_impl->_rs_field = std::make_unique<schifra::galois::field>(field_descriptor,
                                                                                 schifra::galois::primitive_polynomial_size01,
                                                                                 schifra::galois::primitive_polynomial01);

        rs_impl->_rs_field_poly = std::make_unique<schifra::galois::field_polynomial>(*rs_impl->_rs_field);

        if (
                !schifra::make_sequential_root_generator_polynomial(*rs_impl->_rs_field,
                                                                    generator_polynomial_index,
                                                                    generator_polynomial_root_count,
                                                                    *rs_impl->_rs_field_poly)
                )
        {
            throw std::runtime_error{"failed to create sequential root generator"};
        }

        /* Instantiate Finite Field and Generator Polynomials */

        rs_impl->_rs_encoder = std::make_unique<ReedSolomon::rs_encoder_t>(*rs_impl->_rs_field, *rs_impl->_rs_field_poly);
        rs_impl->_rs_decoder = std::make_unique<ReedSolomon::rs_decoder_t>(*rs_impl->_rs_field, generator_polynomial_index);
    }

    return *rs_impl;
}

std::array<int8_t, rs_code_length> rs_encode_block(const char *data, int size) {
    static ReedSolomon::rs_block_t block;


    if (size > rs_data_length) {
        throw std::runtime_error{"invalid data size"};
    }
    std::copy(data, data + size, block.data);
    for(auto j = size; j < rs_data_length; ++j) {
        block.data[j] = 0x00;
    }

    // start encoding
    if(!get_rs_impl()._rs_encoder->encode(block)) {
        throw std::runtime_error{format("an error occurred during reed solomon encode: {}", block.error_as_string())};
    }

    std::array<int8_t, rs_code_length> ret;

    for(auto j = 0; j < rs_data_length; ++j) {
        ret[j] = block.data[j] & 0xFF;
    }

    for(auto j = 0; j < rs_fec_length; ++j) {
        ret[j+rs_data_length] = block.fec(j) & 0xFF;
    }

    return ret;
}

std::array<int8_t, rs_data_length> rs_decode_block(const char *data) {
    static ReedSolomon::rs_block_t block;
    std::copy(data, data+rs_code_length, block.data);

    if(!get_rs_impl()._rs_decoder->decode(block)) {
        throw std::runtime_error{format("an error occurred during reed solomon decode: {}", block.error_as_string())};
    }

    std::array<int8_t, rs_data_length> ret;
    std::copy(block.data, block.data + rs_data_length, ret.data());
    return ret;
}

std::vector<uint16_t> rs_decode_metadata(const DataEmbed::encoded_data_t &enc_data) {
    // decode metadata blocks
    constexpr const int metadata_size = 6;
    std::vector<char> metadata_buf;
    metadata_buf.reserve(2 * metadata_size);
    std::array<char, rs_code_length> code_buf;



    int decoded_size = 0, block_id = 0;
    while(decoded_size < metadata_size) {
        const int s = block_id * rs_code_length;
        if (s + rs_code_length > enc_data.size()) {
            throw std::runtime_error{"end of encoded data reached before fully decoding the data"};
        }
        std::copy(enc_data.data() + s, enc_data.data() + s + rs_code_length, code_buf.data());
        auto decoded_buf = rs_decode_block(code_buf.data());
        std::copy(decoded_buf.begin(), decoded_buf.end(), std::back_inserter(metadata_buf));

        decoded_size += rs_data_length;
        ++block_id;
    }

    return decode_metadata(metadata_buf.data(), metadata_size);
}



DataEmbed::DataEmbed(int block_size, int num_rows, int image_width) : _block_size(block_size),
                                                                                        _num_rows(num_rows),
                                                                                        _image_width(image_width) {

    if (_block_size < 1) {
        throw std::runtime_error{format("block size should be at least 2")};
    }

    if (_block_size % 2 != 0) {
        throw std::runtime_error{format("block size should be even")};
    }


    if (_num_rows < 4) {
        throw std::runtime_error{format("number of data embed rows should be at least 4")};
    }

    _fiducial_marker_size = 4 *  _block_size;

    auto min_image_width = (rs_code_length / 3) * block_size + 2 * _fiducial_marker_size + 2 * _block_size;
    if (_image_width < min_image_width) {
        throw std::runtime_error{format("image width must be at least {}", min_image_width)};
    }

    _image_width_with_marker = _image_width + 5 * _block_size;
    // calculate the amount of bytes that can be represented
    _num_bits_per_block = 24;
    _num_blocks_per_row = (_image_width_with_marker - 2 * _fiducial_marker_size - 2 * _block_size) / block_size;
    _num_bits_per_row = _num_blocks_per_row * _num_bits_per_block;
    _fiducial_marker_col_2 = _fiducial_marker_size + _block_size + _num_blocks_per_row * _block_size + _block_size/2;

    _num_bytes_per_row = _num_bits_per_row / 8;
    _num_bytes_total = _num_bytes_per_row * _num_rows;
    _num_bits_total = _num_bytes_total * 8;

}


DataEmbed::encoded_data_t DataEmbed::encode_data(const std::string &data) const {
    std::stringstream zsstr;
    // compress the data
    zstr::ostream zos(zsstr);
    zos << data;
    zos.flush();
    zsstr.flush();
    auto compressed_data = zsstr.str();

    std::string new_data = encode_metadata({(uint16_t)_num_rows,
                                                 (uint16_t)_num_blocks_per_row,
                                                 (uint16_t)compressed_data.size()}) + compressed_data;

    // reed solomon code
    std::array<char, rs_data_length> data_buf;
    encoded_data_t rs_data;
    rs_data.reserve((new_data.size() / rs_data_length + 1) * rs_code_length);

    for(auto i = 0; i < new_data.size(); i += rs_data_length) {
        int r = std::min(new_data.size(), i + rs_data_length);

        std::copy(new_data.data() + i, new_data.data() + r, data_buf.data());

        auto encoded_block = rs_encode_block(data_buf.data(), r - i);
        // readout
        std::copy(encoded_block.begin(), encoded_block.end(), std::back_inserter(rs_data));
    }

    // expand the data
    auto expanded_data = expand_representation(rs_data, data_embed_expansion);

    if (expanded_data.size() > _num_bytes_total) {
        throw std::runtime_error{format("can only embed {} bytes of data, but the data to embed has {} bytes",
                                        _num_bytes_total, expanded_data.size())};
    }


    // build the row buffer
    encoded_data_t ret(_num_bytes_total, 0x00);

    // copy the data
    std::copy(expanded_data.begin(), expanded_data.end(), ret.begin());

    // expand the data


    return ret;
}

std::string DataEmbed::decode_data(const DataEmbed::encoded_data_t &enc_data) {

    std::vector<int8_t> decoded_data;

    // shrink data
    auto shrunk_data = shrink_representation(enc_data, data_embed_expansion);

    // decode metadata
    auto metadata = rs_decode_metadata(shrunk_data);
    int metadata_size = metadata.size() * sizeof(decltype(metadata)::value_type);
    int original_data_len = metadata[2] + metadata_size;

    // start decoding
    std::array<char, rs_code_length> code_buf;
    auto enc_ptr = 0;
    while(decoded_data.size() < original_data_len) {
        if(enc_ptr + rs_code_length > shrunk_data.size()) {
            throw std::runtime_error{"end of encoded data reached before fully decoding the data"};
        }

        std::copy(shrunk_data.data() + enc_ptr, shrunk_data.data() + enc_ptr + rs_code_length, code_buf.data());
        auto decoded_buf = rs_decode_block(code_buf.data());
        std::copy(decoded_buf.begin(), decoded_buf.end(), std::back_inserter(decoded_data));

        enc_ptr += rs_code_length;
    }


    if (decoded_data.size() < original_data_len) {
        throw std::runtime_error{"cannot correctly decode the data: decoded size mismatch"};
    }

    auto decoded_compressed_data = std::string(decoded_data.begin() + metadata_size, decoded_data.begin() + original_data_len);


    std::stringstream zsstr;
    zsstr << decoded_compressed_data;
    zsstr.flush();
    // compress the data
    zstr::istream zis(zsstr);

    std::string ret(std::istreambuf_iterator<std::string::value_type>(zis), {});

    return ret;
}

cv::Mat DataEmbed::encoded_data_as_image(const cv::Mat &img, const std::string &data) const {
    if(img.cols != _image_width) {
        throw std::runtime_error{format("expected {} cols in the image, get {} instead", _image_width, img.cols)};
    }

    auto encoded_data_buffer = encode_data(data);

    cv::Mat ret(_num_rows * _block_size, _image_width_with_marker, CV_8UC3);
    ret.setTo(cv::Vec3b(255,255,255));


    // build markers
    auto aruco_dict = cv::aruco::getPredefinedDictionary(cv_aruco_marker_dict);
    cv::Mat marker;
    cv::aruco::generateImageMarker(aruco_dict, cv_aruco_marker_inds[0], _fiducial_marker_size, marker);
    cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
    marker.copyTo(ret(cv::Rect(_block_size/2, 0, _fiducial_marker_size, _fiducial_marker_size)));
    cv::aruco::generateImageMarker(aruco_dict, cv_aruco_marker_inds[1], _fiducial_marker_size, marker);
    cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
    marker.copyTo(ret(cv::Rect(_fiducial_marker_col_2, (_num_rows - 4) * _block_size, _fiducial_marker_size, _fiducial_marker_size)));


//    cv::aruco::generateImageMarker(aruco_dict, cv_aruco_marker_inds[2], _fiducial_marker_size, marker);
//    cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
//    marker.copyTo(ret(cv::Rect(_image_width + _block_size/2, 0, _fiducial_marker_size, _fiducial_marker_size)));

    for(auto i = 0; i < _num_rows; ++i) {
        const auto j0 = i * _num_bytes_per_row;
        const auto j1 = (i + 1) * _num_bytes_per_row;
        for(auto j = j0; j < j1; j += 3) {
            for(auto k = 0; k < _block_size; ++k) {
                auto &pix = ret.at<cv::Vec3b>(i * _block_size, _block_size * (j - j0) / 3 + k + _fiducial_marker_size + _block_size);
                pix[0] = encoded_data_buffer[j];
                pix[1] = encoded_data_buffer[j+1];
                pix[2] = encoded_data_buffer[j+2];
            }
        }
    }


    // copy the rows to form blocks
    for(auto i = 0; i < _num_rows; ++i) {
        auto data_region = ret(cv::Range(i * _block_size, i * _block_size + 1),
                               cv::Range(_fiducial_marker_size + _block_size, _fiducial_marker_col_2 - _block_size / 2));
        for(auto j = 1; j < _block_size; ++j) {
            data_region.copyTo(ret(cv::Range(i * _block_size + j, i * _block_size + j +1),
                                   cv::Range(_fiducial_marker_size + _block_size, _fiducial_marker_col_2 - _block_size / 2)));
        }
    }

    // generate the right padder
    cv::Mat padder_v(img.rows, _image_width_with_marker - _image_width, CV_8UC3);
    padder_v.setTo(cv::Vec3b(255, 255, 255));

    cv::aruco::generateImageMarker(aruco_dict, cv_aruco_marker_inds[2], _fiducial_marker_size, marker);
    cv::cvtColor(marker, marker, cv::COLOR_GRAY2BGR);
    marker.copyTo(padder_v(cv::Rect(_block_size/2, 0, _fiducial_marker_size, _fiducial_marker_size)));

    cv::Mat img_vpad;
    cv::hconcat(img, padder_v, img_vpad);

    cv::Mat padder_h(_block_size / 2, _image_width_with_marker, CV_8UC3);
    padder_h.setTo(cv::Vec3b(255, 255, 255));

    // the top pad needs to be aligned with block size to avoid significant quality loss
    cv::Mat padder_16(16, _image_width_with_marker, CV_8UC3);
    padder_16.setTo(cv::Vec3b(255, 255, 255));

    cv::vconcat(padder_16, img_vpad, img_vpad);
    cv::vconcat(img_vpad, padder_h, img_vpad);
    cv::vconcat(img_vpad, ret, ret);
    cv::vconcat(ret, padder_h, ret);

    return ret;
}

cv::Mat DataEmbed::encode_no_data(const cv::Mat &img) const {
    if(img.cols != _image_width) {
        throw std::runtime_error{format("expected {} cols in the image, get {} instead", _image_width, img.cols)};
    }

    cv::Mat ret(_num_rows * _block_size, _image_width_with_marker, CV_8UC3);
    ret.setTo(cv::Vec3b(255,255,255));

    // generate the right padder
    cv::Mat padder_v(img.rows, _image_width_with_marker - _image_width, CV_8UC3);
    padder_v.setTo(cv::Vec3b(255, 255, 255));

    cv::Mat img_vpad;
    cv::hconcat(img, padder_v, img_vpad);

    cv::Mat padder_h(_block_size / 2, _image_width_with_marker, CV_8UC3);
    padder_h.setTo(cv::Vec3b(255, 255, 255));

    // the top pad needs to be aligned with block size to avoid significant quality loss
    cv::Mat padder_16(16, _image_width_with_marker, CV_8UC3);
    padder_16.setTo(cv::Vec3b(255, 255, 255));

    cv::vconcat(padder_16, img_vpad, img_vpad);
    cv::vconcat(img_vpad, padder_h, img_vpad);
    cv::vconcat(img_vpad, ret, ret);
    cv::vconcat(ret, padder_h, ret);

    return ret;
}

size_t DataEmbed::get_data_region_width() const {
    return _num_blocks_per_row * _block_size;
}

size_t DataEmbed::get_data_region_height() const {
    return _num_rows * _block_size;
}


DataEmbed::encoded_data_t expand_representation(const DataEmbed::encoded_data_t &enc_data, int expansion) {
    if(expansion < 1 || expansion >= 8 || 8 % expansion != 0) {
        throw std::runtime_error{"invalid expansion value"};
    }
    int num_bits_per_part = 8 / expansion;
    int num_values_per_part = (1 << num_bits_per_part);
    float image_value_step_size = 256.0f / num_values_per_part;

    std::vector<uint8_t> part_value_lut(num_values_per_part, 0);
    for(auto i = 0; i < part_value_lut.size(); ++i) {
        part_value_lut[i] = lround((i  + 0.5f) * image_value_step_size);
    }


    uint8_t base_extractor = num_values_per_part - 1;
    std::vector<uint8_t> part_extractors(expansion, 0);
    for(auto i = 0; i < part_extractors.size(); ++i) {
        part_extractors[i] = base_extractor << (i * num_bits_per_part);
    }

    DataEmbed::encoded_data_t ret;
    ret.reserve(enc_data.size() * expansion);

    for(auto i = 0; i < enc_data.size(); ++i) {
        for(auto j = 0; j < part_extractors.size(); ++j) {
            uint8_t part = (enc_data[i] &  part_extractors[j]) >> (j * num_bits_per_part);
            ret.push_back(part_value_lut[part]);
        }
    }

    return ret;
}

DataEmbed::encoded_data_t shrink_representation(const DataEmbed::encoded_data_t &enc_data, int expansion) {
    if(expansion < 1 || expansion >= 8 || 8 % expansion != 0) {
        throw std::runtime_error{"invalid expansion value"};
    }
    if(enc_data.size() % expansion != 0) {
        throw std::runtime_error{"the expanded representation to be decoded has invalid size"};
    }

    int num_bits_per_part = 8 / expansion;
    int num_values_per_part = (1 << num_bits_per_part);
    float image_value_step_size = 256.0f / num_values_per_part;

    std::vector<uint8_t> part_value_lut(num_values_per_part, 0);
    for(auto i = 0; i < part_value_lut.size(); ++i) {
        part_value_lut[i] = lround((i  + 0.5f) * image_value_step_size);
    }


    uint8_t base_extractor = num_values_per_part - 1;
    std::vector<uint8_t> part_extractors(expansion, 0);
    for(auto i = 0; i < part_extractors.size(); ++i) {
        part_extractors[i] = base_extractor << (i * num_bits_per_part);
    }

    DataEmbed::encoded_data_t data_closest;
    data_closest.reserve(enc_data.size());
    std::vector<int16_t> dist(num_values_per_part, 0);

    for(auto i = 0; i < enc_data.size(); ++i) {
        // find its original representation
        for(auto j = 0; j < num_values_per_part; ++j) {
            dist[j] = abs((int16_t)part_value_lut[j] - (int16_t)enc_data[i]);
        }
        auto min_ele = std::min_element(dist.begin(), dist.end());
        auto min_ele_ind = std::distance(dist.begin(), min_ele);
        data_closest.push_back(min_ele_ind);
    }

    DataEmbed::encoded_data_t ret;
    ret.reserve(enc_data.size()/expansion);
    for(auto i = 0; i < data_closest.size(); i += expansion) {
        uint8_t val = 0;
        for(auto j = 0; j < expansion; ++j) {
            uint8_t part_val = data_closest[i+j] << (j * num_bits_per_part);
            val = val | part_val;
        }
        ret.push_back(val);
    }

    return ret;
}
