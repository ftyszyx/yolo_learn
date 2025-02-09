#include <iostream>
#include <vector>
#include <getopt.h>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/mnt/d/work/github/yolo_learn/examples/YOLOv8-CPP-Inference/"; // Set your ultralytics base path
    std::filesystem::path result_path = projectBasePath + "/result";
    if (!std::filesystem::exists(result_path))
    {
        std::filesystem::create_directory(result_path);
    }
    bool runOnGPU = false;
    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolov8n.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);
    std::filesystem::path imgs_path = projectBasePath + "/images";
    std::vector<std::string> imageNames;
    for (auto &i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
        {
            std::string img_path = i.path().string();
            cv::Mat frame = cv::imread(img_path);
            // Inference starts here...
            std::vector<Detection> output = inf.runInference(frame);

            int detections = output.size();
            std::cout << "Number of detections:" << detections << std::endl;

            for (int i = 0; i < detections; ++i)
            {
                Detection detection = output[i];

                cv::Rect box = detection.box;
                cv::Scalar color = detection.color;

                // Detection box
                cv::rectangle(frame, box, color, 2);

                // Detection box text
                std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                cv::rectangle(frame, textBox, color, cv::FILLED);
                cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            }
            // Inference ends here...

            // This is only for preview purposes
            float scale = 0.8;
            cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
            std::string img_name = img_path.substr(img_path.find_last_of("/") + 1);
            // save the image
            cv::imwrite(result_path / img_name, frame);
        }
    }
}
