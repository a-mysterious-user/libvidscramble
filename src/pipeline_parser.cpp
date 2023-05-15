#include "pipeline_parser.h"


std::shared_ptr<VideoScramblePipeline> build_pipeline_from_json(const std::string &json_str) {
    nlohmann::json json_data = nlohmann::json::parse(json_str);

    if (json_data.find("steps") == json_data.end()) {
        throw std::runtime_error{"expected \"steps\" key in the json string"};
    }

    auto scrambler_steps = std::make_shared<std::vector<pipeline_step_t>>();

    const auto &steps = json_data["steps"];
    for (auto step_iter = steps.begin(); step_iter != steps.end(); ++step_iter) {
        const auto &step = *step_iter;
        auto step_name = step["name"].get<std::string>();

        pipeline_step_t new_step;

        if(step_name == std::string{"RowShuffle"}) {
            new_step = construct_scrambler(step, RowShuffle_cspec);
        } else if (step_name == std::string{"ImageTranspose"}) {
            new_step = construct_scrambler(step, ImageTranspose_cspec);
        }  else if (step_name == std::string{"RowMix"}) {
            new_step = construct_scrambler(step, RowMix_cspec);
        } else if (step_name == std::string{"ImageShift"}) {
            new_step = construct_scrambler(step, ImageShift_cspec);
        } else {
            throw std::runtime_error{format("unknown scrambler method \"{}\"", step_name)};
        }

        scrambler_steps->push_back(new_step);
    }


    auto ret = std::make_shared<VideoScramblePipeline>(scrambler_steps,
                                                   json_data["data_embed_block_size"].get<int>(),
                                                   json_data["data_embed_num_rows"].get<int>()
                                                   );

    auto embed_interval_find = json_data.find("data_embed_interval");
    if (embed_interval_find != json_data.end()) {
        ret->set_data_embed_interval(embed_interval_find->get<int>());
    }

    return ret;
}
