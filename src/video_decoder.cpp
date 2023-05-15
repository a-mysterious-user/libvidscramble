#include "pipeline_parser.h"
#include <argparse/argparse.hpp>


int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("video_decoder");

    program.add_argument("video_filename");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << format("invalid arguments: {}", err.what());
        return 1;
    }

    auto video_filename = program.get<std::string>("video_filename");

    cv::VideoCapture cap(video_filename);

    if(!cap.isOpened()) {
        std::cerr << format("error opening video file \"{}\"", video_filename);
    }

    ImageDataTransform tf;
    auto data_ex_tf_success = false;
    int frame_id = 0;
    std::shared_ptr<VideoScramblePipeline> pipeline;

    while(true){

        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty()){
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);


        if (!data_ex_tf_success) {
            data_ex_tf_success = VideoScramblePipeline::get_data_extraction_transform(frame, tf);
            if (!data_ex_tf_success) {
                std::cout << format("failed to extract data transformation information in frame {}\n", frame_id++);
                continue;
            } else {
                // decode data
                auto data = VideoScramblePipeline::extract_data(frame, tf);
                std::cout << format("decoded data from the video: {}", data);
                auto data_json = nlohmann::json::parse(data);
                // build pipeline
                pipeline = build_pipeline_from_json(data);
                // create a duumy image to fit the pipeline
                cv::Mat dummy(data_json["state"]["input_height"].get<int>(), data_json["state"]["input_width"].get<int>(), CV_8UC3);
                pipeline->fit(dummy);
                // sync state
                pipeline->sync_state(data_json["state"]);
            }
        }

        // apply inverse transform
        auto new_frame = pipeline->inverse_transform(frame, tf);
        cv::cvtColor(new_frame, new_frame, cv::COLOR_RGB2BGR);

        // Display the resulting frame
        imshow( format("video frame", frame_id), new_frame);

        // Press  ESC on keyboard to exit
        auto c = (char)cv::waitKey(10);
        if(c == 27){
            break;
        }

        ++frame_id;
    }

    cap.release();
    cv::destroyAllWindows();


    return 0;
}
