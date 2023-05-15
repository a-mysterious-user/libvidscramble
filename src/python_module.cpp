#include <pybind11/pybind11.h>
#include "pipeline.h"
#include "pipeline_parser.h"
#include "ndarray_converter.h"

namespace py = pybind11;


PYBIND11_MODULE(py_vidscramble, m) {

    NDArrayConverter::init_numpy();

    py::class_<VideoScramblePipeline, std::shared_ptr<VideoScramblePipeline>>(m, "VideoScramblePipeline")
        .def(py::init<std::shared_ptr<std::vector<pipeline_step_t>>, int, int>())
        .def("fit", &VideoScramblePipeline::fit)
        .def("transform", &VideoScramblePipeline::transform)
        .def("inverse_transform", &VideoScramblePipeline::inverse_transform)
        .def("reset_timestamp", &VideoScramblePipeline::reset_timestamp)
        .def("set_timestamp_increment", &VideoScramblePipeline::set_timestamp_increment)
        .def("increment_timestamp", &VideoScramblePipeline::increment_timestamp)
        .def("to_json", &VideoScramblePipeline::to_json)
        .def("to_json_image", py::overload_cast<>(&VideoScramblePipeline::to_json_image, py::const_))
        .def("to_json_image", py::overload_cast<const cv::Mat&>(&VideoScramblePipeline::to_json_image, py::const_))
        .def("to_no_data_image", &VideoScramblePipeline::to_no_data_image)
        .def("get_data_extraction_transform", &VideoScramblePipeline::get_data_extraction_transform)
        .def("extract_data", &VideoScramblePipeline::extract_data)
        .def("set_data_embed_interval", &VideoScramblePipeline::set_data_embed_interval)
        .def("get_data_embed_interval", &VideoScramblePipeline::get_data_embed_interval);


    py::class_<ImageDataTransform>(m, "ImageRecoveryInfo")
        .def(py::init<>())
        .def_readwrite("data_region_x", &ImageDataTransform::data_region_x)
        .def_readwrite("data_region_y", &ImageDataTransform::data_region_y)
        .def_readwrite("data_region_height", &ImageDataTransform::data_region_height)
        .def_readwrite("data_region_width", &ImageDataTransform::data_region_width)
        .def_readwrite("num_data_rows", &ImageDataTransform::num_data_rows)
        .def_readwrite("num_data_cols", &ImageDataTransform::num_data_cols)
        .def_readwrite("original_image_region_height", &ImageDataTransform::original_image_region_height)
        .def_readwrite("original_image_region_width", &ImageDataTransform::original_image_region_width)
        .def_readwrite("original_data_region_width", &ImageDataTransform::original_data_region_width)
        .def_readwrite("original_data_region_height", &ImageDataTransform::original_data_region_height);


    m.def("build_pipeline_from_json", &build_pipeline_from_json);
}