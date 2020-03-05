#include "HeadPoseDetectorMNN.h"

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/ExprCreator.hpp>

#define kInputSize 224
#define kOutputSize 66

using namespace MNN::Express;

HeadPoseDetectorMNN::HeadPoseDetectorMNN()
{
    //m_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile("hopenet_lite.mnn"));
    m_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile("hopenet_lite_quantized.mnn"));

    MNN::ScheduleConfig config;
    config.numThread = 1;
    config.type = MNN_FORWARD_METAL;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
    config.backendConfig = &backendConfig;

    m_session = m_interpreter->createSession(config);

    m_tensor = m_interpreter->getSessionInput(m_session, nullptr);

}

HeadPoseDetectorMNN::~HeadPoseDetectorMNN()
{
    m_interpreter->releaseModel();
    m_interpreter->releaseSession(m_session);
}

bool HeadPoseDetectorMNN::detect(const cv::Mat &rgb, /*out*/double &yaw, /*out*/double &pitch, /*out*/double &roll)
{
    if (rgb.empty())
        return false;

    m_interpreter->resizeTensor(m_tensor, {1, 3, kInputSize, kInputSize});
    m_interpreter->resizeSession(m_session);

    const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
                MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, mean_vals, 3, norm_vals, 3));

    const int imgWidth = rgb.cols;
    const int imgHeight = rgb.rows;

    MNN::CV::Matrix trans;
    trans.setIdentity();
    trans.postScale(1.0f / imgWidth, 1.0f / imgHeight);
    trans.postScale(kInputSize, kInputSize);
    trans.invert(&trans);

    pretreat->setMatrix(trans);
    pretreat->convert(rgb.data, imgWidth, imgHeight, rgb.step[0], m_tensor);

    m_interpreter->runSession(m_session);

    const MNN::Tensor *yawTensor = m_interpreter->getSessionOutput(m_session, "616");
    const MNN::Tensor *pitchTensor = m_interpreter->getSessionOutput(m_session, "617");
    const MNN::Tensor *rollTensor = m_interpreter->getSessionOutput(m_session, "618");

    yaw = calcPoseValue(yawTensor);
    pitch = calcPoseValue(pitchTensor);
    roll = calcPoseValue(rollTensor);

    return true;
}

double HeadPoseDetectorMNN::calcPoseValue(const MNN::Tensor *tensor)
{
    if (tensor == nullptr)
        return 0;

    tensor->getDimensionType();

    MNN::Tensor tensorHost(tensor, tensor->getDimensionType());

    tensor->copyToHostTensor(&tensorHost);

    std::vector<float> vals;

    for (int i = 0; i < kOutputSize; i++)
    {
        vals.push_back(tensorHost.host<float>()[i]);
    }

    auto input = _Input({1, 66}, NCHW);
    auto inputPtr = input->writeMap<float>();
    memcpy(inputPtr, vals.data(), vals.size() * sizeof(float));
    input->unMap();
    auto output = _Softmax(input);
    auto predicted = output->readMap<float>();

    double result = 0;
    for (int i = 0; i < kOutputSize; i++)
    {
        result += (predicted[i] * i);
    }

    return result * 3 - 99;
}
