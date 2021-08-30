#ifndef CTENSORFLOW_H
#define CTENSORFLOW_H

///***********************************///
///* C TENSORFLOW LIBRARY v1         *///
///* Author : razr191120             *///
///* Date : 18 Nov                   *///
///***********************************///

//// OPENCV LIBS ////
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//// QT CORE LIBS ////
#include <QMap>
#include <QImage>
#include <QList>
#include <QtCore>
#include <QDebug>
#include <QByteArray>
#include <QBuffer>
#include <QString>

//// STD LIBS ////
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stack>
#include <string>
#include <codecvt>
#include <fstream>

//// TENSORFLOW LIBS ////
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>
#include <scope_guard.hpp>

namespace CTensorFlow
{

/* NAMESPACE FOR HEADER FOR TensorFlow Utility Functions USING TENSORFLOW CAPI */
//// ****DO NOT EDIT THIS NAMESPACE**** ////
namespace TensorFlowUtils
{

static void DeallocateBuffer(void* data, size_t) {
    std::free(data);
}

/* Tensor Info Functions */
void PrintInputs(TF_Graph*, TF_Operation* op); // Print Input Layer Information
void PrintOutputs(TF_Graph* graph, TF_Operation* op, TF_Status* status); // Print Output Layer Information
void PrintTensorInfo(TF_Graph* graph, const char* layer_name, TF_Status* status); // Print Graph Information


/* TensorFlow Utility Functions , Do not Edit */
TF_Graph* LoadGraph(const char* graph_path, const char* checkpoint_prefix, TF_Status* status = nullptr);
TF_Graph* LoadGraph(const char* graph_path, TF_Status* status = nullptr);
void DeleteGraph(TF_Graph* graph);
TF_Session* CreateSession(TF_Graph* graph, TF_SessionOptions* options, TF_Status* status = nullptr);
TF_Session* CreateSession(TF_Graph* graph, TF_Status* status = nullptr);
TF_Code DeleteSession(TF_Session* session, TF_Status* status = nullptr);
TF_Code RunSession(TF_Session* session,
                   const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                   const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                   TF_Status* status = nullptr);
TF_Code RunSession(TF_Session* session,
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   TF_Status* status = nullptr);
TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len);
TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len = 0);
TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len = 0);
void DeleteTensor(TF_Tensor* tensor);
void DeleteTensors(const std::vector<TF_Tensor*>& tensors);
bool SetTensorData(TF_Tensor* tensor, const void* data, std::size_t len);
std::vector<std::int64_t> GetTensorShape(TF_Graph* graph, const TF_Output& output);
std::vector<std::vector<std::int64_t>> GetTensorsShape(TF_Graph* graph, const std::vector<TF_Output>& output);
TF_SessionOptions* CreateSessionOptions(double gpu_memory_fraction, TF_Status* status = nullptr);
TF_SessionOptions* CreateSessionOptions(std::uint8_t intra_op_parallelism_threads, std::uint8_t inter_op_parallelism_threads, TF_Status* status = nullptr);
void DeleteSessionOptions(TF_SessionOptions* options);
const char* DataTypeToString(TF_DataType data_type);
const char* CodeToString(TF_Code code);
TF_Buffer* ReadBufferFromFile(const char* file);
TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status);

template <typename T>
TF_Tensor* CreateTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, const std::vector<T>& data) {
    return CreateTensor(data_type,
                        dims.data(), dims.size(),
                        data.data(), data.size() * sizeof(T));
}

template <typename T>
void SetTensorData(TF_Tensor* tensor, const std::vector<T>& data) {
    SetTensorsData(tensor, data.data(), data.size() * sizeof(T));
}

template <typename T>
std::vector<T> GetTensorData(const TF_Tensor* tensor) {
    if (tensor == nullptr) {
        return {};
    }
    auto data = static_cast<T*>(TF_TensorData(tensor));
    auto size = TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor));
    if (data == nullptr || size <= 0) {
        return {};
    }

    return {data, data + size};
}

template <typename T>
std::vector<std::vector<T>> GetTensorsData(const std::vector<TF_Tensor*>& tensors) {
    std::vector<std::vector<T>> data;
    data.reserve(tensors.size());
    for (auto t : tensors) {
        data.push_back(GetTensorData<T>(t));
    }

    return data;
}
}
//// ****DO NOT EDIT THIS NAMESPACE**** ////
/* NAMESPACE TensorFlow Utility ENDS */

/* NAMESPACE FOR HEADER FOR Classification USING TENSORFLOW CAPI */
namespace Classification
{
using namespace TensorFlowUtils;

class CAPITfClassification
{
public:
    CAPITfClassification();
    // Graph PB Path
    TF_Graph *graphptr;

    // Input Layer Operand
    TF_Output input_op;

    // Output Layer Operand
    TF_Output output_op;

    // TF Status Operand
    TF_Status *status;

    // TF Options Operand
    TF_SessionOptions *options;

    // TF Session Var
    TF_Session* sess;

    // Image Path As Recieved From CProcess
    QString _ImageName;
    QString ImageName() const;
    void setImageName(const QString &ImageName);

    // Model Path
    std::string _ModelName;
    std::string ModelName() const;
    void setModelName(const std::string &ModelName);

    // Input Layer
    std::string _InputLayer;
    std::string InputLayer() const;
    void setInputLayer(const std::string &InputLayer);

    // Output Layer
    std::string _OutputLayer;
    std::string OutputLayer() const;
    void setOutputLayer(const std::string &OutputLayer);

    // Image Width
    int _ImageWidth;
    int ImageWidth() const;
    void setImageWidth(int ImageWidth);

    // Image Height
    int _ImageHeight;
    int ImageHeight() const;
    void setImageHeight(int ImageHeight);

    // Number of Channels In Output
    int _Channel;
    int Channel() const;
    void setChannel(int Channel);

    // TensorGraph Dimensions
    std::vector<std::int64_t> _Input_dims;
    std::vector<std::int64_t> Input_dims() const;
    void setInput_dims(const std::vector<std::int64_t> &Input_dims);

    // Load Graph and Vars
    void Init(std::string ModelName,
              std::string InputLayer,
              std::string OutputLayer,
              int ImageWidth,
              int ImageHeight,
              int Channel
              );

    // Pre-Pocessing
    std::vector<float> ConvertQImageToFloat(QImage PrepImage);
    //std::vector<float> ConvertCVMatToFloat(cv::Mat PrepImage);

    // Inference
    float* RunInferClassification(QImage InferImage);
    //float* RunInferClassification(cv::Mat InferImage);

    // Exit
    void DeleteTensorVars();
};

}
/* NAMESPACE Classification ENDS */

/* NAMESPACE FOR HEADER FOR Segmentation USING TENSORFLOW CAPI */
namespace Segmentation
{

using namespace TensorFlowUtils;

class CAPITfSegmentation
{
public:

    // Class CTor
    CAPITfSegmentation();

    // Graph PB Path
    TF_Graph *graphptr;

    // Input Layer Operand
    TF_Output input_op;

    // Output Layer Operand
    TF_Output output_op;

    // TF Status Operand
    TF_Status *status;

    // TF Options Operand
    TF_SessionOptions *options;

    // TF Session Var
    TF_Session* sess;

    // Image Path As Recieved From CProcess
    QString _ImageName;
    QString ImageName() const;
    void setImageName(const QString &ImageName);

    // Model Path
    std::string _ModelName;
    std::string ModelName() const;
    void setModelName(const std::string &ModelName);

    // Input Layer
    std::string _InputLayer;
    std::string InputLayer() const;
    void setInputLayer(const std::string &InputLayer);

    // Output Layer
    std::string _OutputLayer;
    std::string OutputLayer() const;
    void setOutputLayer(const std::string &OutputLayer);

    // Image Width
    int _ImageWidth;
    int ImageWidth() const;
    void setImageWidth(int ImageWidth);

    // Image Height
    int _ImageHeight;
    int ImageHeight() const;
    void setImageHeight(int ImageHeight);

    // Number of Channels In Output
    int _Channel;
    int Channel() const;
    void setChannel(int Channel);

    // TensorGraph Dimensions
    std::vector<std::int64_t> _Input_dims;
    std::vector<std::int64_t> Input_dims() const;
    void setInput_dims(const std::vector<std::int64_t> &Input_dims);

    // Load Graph and Vars
    void Init(std::string ModelName,
              std::string InputLayer,
              std::string OutputLayer,
              int ImageWidth,
              int ImageHeight,
              int Channel
              );

    // Pre-Pocessing
    std::vector<float> ConvertQImageToFloat(QImage PrepImage);
    //std::vector<float> ConvertCVMatToFloat(cv::Mat PrepImage);

    // Inference
    QMap<QString,QImage> RunInferSegmentation(QImage InferImage);
    //QMap<QString,QImage>RunInferSegmentation(cv::Mat InferImage);

    // Post-Processing Post Segmentataion
    QMap<QString,QImage> ConvertFloatToQImage(float* outputTensor);

    // Exit
    void DeleteTensorVars();
};

}
/* NAMESPACE Segmentation ENDS */

/* NAMESPACE FOR HEADER FOR Segmentation_PostProcessing USING OPENCV API */
namespace Segmentation_PostProcessing
{
using namespace std;
using namespace cv;
class CAPISegmentation
{
public:
    CAPISegmentation();
    std::vector<std::vector<cv::Point> > contours;
    QMap<QString, vector<vector<Point> > > postProcess(QMap<QString,QImage>,int,int,double threshold);
    cv::Mat QImageToCvMat(QImage image);
    vector<cv::Mat>mat_vector;
    cv::Mat contourOutput;
    vector<std::vector<cv::Point>> find_Contours(cv::Mat,double);
    int Id=0;
    int display_Id=0;
};
}
/* NAMESPACE Segmentation_PostProcessing ENDS */

/* NAMESPACE FOR HEADER FOR ObjectDetection USING TENSORFLOW CAPI */
namespace ObjectDetection
{
namespace ssd{}
namespace yolo{}
namespace frcnn{}
}
/* NAMESPACE ObjectDetection USING ENDS */

}

#endif // CTENSORFLOW_H
