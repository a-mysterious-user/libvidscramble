#include "pipeline.h"

int main() {

//    DataEmbed data_embed(8, 4, 1280);
//
//    // std::string random_data{60, 0x02};
//
//    std::string random_data{"As Alexandre C. mentioned, the problem comes down to window's destructor being implicitly defined in places where the type of window_impl is still incomplete. In addition to his solutions, another workaround that I've used is to declare a Deleter functor in the header:"};
//
//    auto encoded_data = data_embed.encode_data(random_data);
//
//    auto decoded_data = data_embed.decode_data(encoded_data);
//
//    std::cout << decoded_data << "\n";

    auto img = cv::imread("../test/test.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

//    cv::imshow("", img);
//    cv::waitKey(0);

    ImageDataTransform tf;
    auto success = VideoScramblePipeline::get_data_extraction_transform(img, tf);

    if(!success) {
        std::cout << "unable to parse data\n";
        exit(0);
    }

    auto data = VideoScramblePipeline::extract_data(img, tf);

    std::cout << data  << "\n";
//
    auto img_region = VideoScramblePipeline::extract_image_region(img, tf);

    cv::cvtColor(img_region, img_region, cv::COLOR_RGB2BGR);

    cv::imshow("", img_region);
    cv::waitKey();
    cv::destroyAllWindows();

//    DataEmbed::encoded_data_t data{};
//    data.push_back(25);
//    data.push_back(30);
//    data.push_back(200);
//    auto new_repr = expand_representation(data, 4);
//    auto new_repr_decode = shrink_representation(new_repr, 4);


//    auto encoded_image = data_embed.encoded_data_as_image(random_data);
//
//    cv::imshow("", encoded_image);
//    cv::waitKey(0);


    return 0;
}