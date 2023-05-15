#pragma once


#include "pipeline.h"
#include "scrambler_constructor_spec.h"


std::shared_ptr<VideoScramblePipeline> build_pipeline_from_json(const std::string &json_str);
