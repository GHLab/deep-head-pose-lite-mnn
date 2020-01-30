#ifndef HEADPOSEDETECTORMNN_H
#define HEADPOSEDETECTORMNN_H

#include <memory>

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <opencv2/opencv.hpp>

class HeadPoseDetectorMNN
{
public:
    HeadPoseDetectorMNN(HeadPoseDetectorMNN const&) = delete;
    HeadPoseDetectorMNN& operator=(HeadPoseDetectorMNN const&) = delete;

    static std::shared_ptr<HeadPoseDetectorMNN> instance()
    {
        static std::shared_ptr<HeadPoseDetectorMNN> s { new HeadPoseDetectorMNN };
        return s;
    }
    ~HeadPoseDetectorMNN();

private:
    HeadPoseDetectorMNN();

public:
    bool detect(const cv::Mat &rgb, /*out*/double &yaw, /*out*/double &pitch, /*out*/double &roll);

private:
    std::shared_ptr<MNN::Interpreter> m_interpreter;
    MNN::Session *m_session = nullptr;
    MNN::Tensor *m_tensor = nullptr;
};

#endif // HEADPOSEDETECTORMNN_H
