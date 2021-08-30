#include "ctensorflow.h"
#include <QDateTime>


namespace CTensorFlow
{

/* NAMESPACE TensorFlow Utility Functions USING TENSORFLOW CAPI **DO NOT EDIT**  */
namespace TensorFlowUtils
{

void PrintInputs(TF_Graph*, TF_Operation* op)
{
    auto num_inputs = TF_OperationNumInputs(op);

    for (auto i = 0; i < num_inputs; ++i) {
        auto input = TF_Input{op, i};
        auto type = TF_OperationInputType(input);
        std::cout << "Input: " << i <<std::endl<< " type: " << DataTypeToString(type) << std::endl;
    }
}

void PrintOutputs(TF_Graph* graph, TF_Operation* op, TF_Status* status)
{
    auto num_outputs = TF_OperationNumOutputs(op);

    for (int i = 0; i < num_outputs; ++i) {
        auto output = TF_Output{op, i};
        auto type = TF_OperationOutputType(output);
        auto num_dims = TF_GraphGetTensorNumDims(graph, output, status);

        if (TF_GetCode(status) != TF_OK) {
            std::cout << "Can't get tensor dimensionality" <<
                         std::to_string(TF_GetCode(status))<<std::endl;

            continue;
        }

        std::cout << " dims: " << num_dims<<std::endl;

        if (num_dims <= 0) {
            std::cout << " []" << std::endl;;
            continue;
        }

        std::vector<std::int64_t> dims(num_dims);

        std::cout << "Output: " << i <<std::endl<< " type: " << DataTypeToString(type)<<std::endl;
        TF_GraphGetTensorShape(graph, output, dims.data(), num_dims, status);

        if (TF_GetCode(status) != TF_OK) {
            std::cout << "Can't get get tensor shape" << std::endl;
            continue;
        }

        std::cout << " [";
        for (auto d = 0; d < num_dims; ++d) {
            std::cout << dims[d];
            if (d < num_dims - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

void PrintTensorInfo(TF_Graph* graph, const char* layer_name, TF_Status* status)
{
    std::cout << "Tensor: " << layer_name<<std::endl;
    auto op = TF_GraphOperationByName(graph, layer_name);

    if (op == nullptr) {
        std::cout << "Could not get " << layer_name << std::endl;
        return;
    }

    auto num_inputs = TF_OperationNumInputs(op);
    auto num_outputs = TF_OperationNumOutputs(op);
    std::cout << " inputs: " << num_inputs << " outputs: " << num_outputs << std::endl;

    PrintInputs(graph, op);
    PrintOutputs(graph, op, status);
}


/** TensorFlow Utility Definitions Do Not Edit Onwards **/


TF_Buffer* ReadBufferFromFile(const char* file) {
    std::ifstream f(file, std::ios::binary);
    SCOPE_EXIT{ f.close(); };
    if (f.fail() || !f.is_open()) {
        return nullptr;
    }

    if (f.seekg(0, std::ios::end).fail()) {
        return nullptr;
    }
    auto fsize = f.tellg();
    if (f.seekg(0, std::ios::beg).fail()) {
        return nullptr;
    }

    if (fsize <= 0) {
        return nullptr;
    }

    auto data = static_cast<char*>(std::malloc(fsize));
    if (f.read(data, fsize).fail()) {
        return nullptr;
    }

    auto buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;

    return buf;
}

TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status) {
    auto str_len = std::strlen(str);
    auto nbytes = 8 + TF_StringEncodedSize(str_len); // 8 extra bytes - for start_offset.
    auto tensor = TF_AllocateTensor(TF_STRING, nullptr, 0, nbytes);
    auto data = static_cast<char*>(TF_TensorData(tensor));
    std::memset(data, 0, 8);
    TF_StringEncode(str, str_len, data + 8, nbytes - 8, status);
    return tensor;
}



TF_Graph* LoadGraph(const char* graph_path, const char* checkpoint_prefix, TF_Status* status) {
    if (graph_path == nullptr) {
        return nullptr;
    }

    auto buffer = ReadBufferFromFile(graph_path);
    if (buffer == nullptr) {
        return nullptr;
    }

    MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
    if (status == nullptr) {
        status = TF_NewStatus();
    } else {
        delete_status.dismiss();
    }

    auto graph = TF_NewGraph();
    auto opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(graph, buffer, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    if (TF_GetCode(status) != TF_OK) {
        DeleteGraph(graph);
        return nullptr;
    }

    if (checkpoint_prefix == nullptr) {
        return graph;
    }

    auto checkpoint_tensor = ScalarStringTensor(checkpoint_prefix, status);
    SCOPE_EXIT{ DeleteTensor(checkpoint_tensor); };
    if (TF_GetCode(status) != TF_OK) {
        DeleteGraph(graph);
        return nullptr;
    }

    auto input = TF_Output{TF_GraphOperationByName(graph, "save/Const"), 0};
    auto restore_op = TF_GraphOperationByName(graph, "save/restore_all");

    auto session = CreateSession(graph);
    SCOPE_EXIT{ DeleteSession(session,nullptr); };
    if (session == nullptr) {
        DeleteGraph(graph);
        return nullptr;
    }

    TF_SessionRun(session,
                  nullptr, // Run options.
                  &input, &checkpoint_tensor, 1, // Input tensors, input tensor values, number of inputs.
                  nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                  &restore_op, 1, // Target operations, number of targets.
                  nullptr, // Run metadata.
                  status // Output status.
                  );

    if (TF_GetCode(status) != TF_OK) {
        DeleteGraph(graph);
        return nullptr;
    }

    return graph;
}

TF_Graph* LoadGraph(const char* graph_path, TF_Status* status) {
    return LoadGraph(graph_path, nullptr, status);
}

void DeleteGraph(TF_Graph* graph) {
    if (graph != nullptr) {
        TF_DeleteGraph(graph);
    }
}

TF_Session* CreateSession(TF_Graph* graph, TF_SessionOptions* options, TF_Status* status) {
    if (graph == nullptr) {
        return nullptr;
    }

    MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
    if (status == nullptr) {
        status = TF_NewStatus();
    } else {
        delete_status.dismiss();
    }

    MAKE_SCOPE_EXIT(delete_options){ DeleteSessionOptions(options);};
    if (options == nullptr) {
        options = TF_NewSessionOptions();
    } else {
        delete_options.dismiss();
    }

    auto session = TF_NewSession(graph, options, status);
    if (TF_GetCode(status) != TF_OK) {
        DeleteSession(session,nullptr);
        return nullptr;
    }

    return session;
}

TF_Session* CreateSession(TF_Graph* graph, TF_Status* status) {
    return CreateSession(graph, nullptr, status);
}

TF_Code DeleteSession(TF_Session* session, TF_Status* status) {
    if (session == nullptr) {
        return TF_INVALID_ARGUMENT;
    }

    MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
    if (status == nullptr) {
        status = TF_NewStatus();
    } else {
        delete_status.dismiss();
    }

    TF_CloseSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
        SCOPE_EXIT{ TF_CloseSession(session, status); };
        SCOPE_EXIT{ TF_DeleteSession(session, status); };
        return TF_GetCode(status);
    }

    TF_DeleteSession(session, status);
    if (TF_GetCode(status) != TF_OK) {
        SCOPE_EXIT{ TF_DeleteSession(session, status); };
        return TF_GetCode(status);
    }

    return TF_OK;
}

TF_Code RunSession(TF_Session* session,
                   const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                   const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs,
                   TF_Status* status) {
    if (session == nullptr ||
            inputs == nullptr || input_tensors == nullptr ||
            outputs == nullptr || output_tensors == nullptr) {
        return TF_INVALID_ARGUMENT;
    }

    MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
    if (status == nullptr) {
        status = TF_NewStatus();
    } else {
        delete_status.dismiss();
    }


    TF_SessionRun(session,
                  nullptr, // Run options.
                  inputs, input_tensors, static_cast<int>(ninputs), // Input tensors, input tensor values, number of inputs.
                  outputs, output_tensors, static_cast<int>(noutputs), // Output tensors, output tensor values, number of outputs.
                  nullptr, 0, // Target operations, number of targets.
                  nullptr, // Run metadata.
                  status // Output status.
                  );

    return TF_GetCode(status);
}

TF_Code RunSession(TF_Session* session,
                   const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                   const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors,
                   TF_Status* status) {
    return RunSession(session,
                      inputs.data(), input_tensors.data(), input_tensors.size(),
                      outputs.data(), output_tensors.data(), output_tensors.size(),
                      status);
}

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len) {
    if (dims == nullptr) {
        return nullptr;
    }

    return TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
}

TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len) {
    return CreateEmptyTensor(data_type, dims.data(), dims.size(), len);
}

TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len) {
    auto tensor = CreateEmptyTensor(data_type, dims, num_dims, len);
    if (tensor == nullptr) {
        return nullptr;
    }

    auto tensor_data = TF_TensorData(tensor);
    if (tensor_data == nullptr) {
        DeleteTensor(tensor);
        return nullptr;
    }

    len = std::min(len, TF_TensorByteSize(tensor));
    if (data != nullptr && len != 0) {
        std::memcpy(tensor_data, data, len);
    }

    return tensor;
}

void DeleteTensor(TF_Tensor* tensor) {
    if (tensor != nullptr) {
        TF_DeleteTensor(tensor);
    }
}

void DeleteTensors(const std::vector<TF_Tensor*>& tensors) {
    for (auto& t : tensors) {
        DeleteTensor(t);
    }
}

bool SetTensorData(TF_Tensor* tensor, const void* data, std::size_t len) {
    auto tensor_data = TF_TensorData(tensor);
    len = std::min(len, TF_TensorByteSize(tensor));
    if (tensor_data != nullptr && data != nullptr && len != 0) {
        std::memcpy(tensor_data, data, len);
        return true;
    }

    return false;
}

std::vector<std::int64_t> GetTensorShape(TF_Graph* graph, const TF_Output& output) {
    auto status = TF_NewStatus();
    SCOPE_EXIT{ TF_DeleteStatus(status); };

    auto num_dims = TF_GraphGetTensorNumDims(graph, output, status);
    if (TF_GetCode(status) != TF_OK) {
        return {};
    }

    std::vector<std::int64_t> result(num_dims);
    TF_GraphGetTensorShape(graph, output, result.data(), num_dims, status);
    if (TF_GetCode(status) != TF_OK) {
        return {};
    }

    return result;
}

std::vector<std::vector<std::int64_t>> GetTensorsShape(TF_Graph* graph, const std::vector<TF_Output>& outputs) {
    std::vector<std::vector<std::int64_t>> result;
    result.reserve(outputs.size());

    for (const auto& o : outputs) {
        result.push_back(GetTensorShape(graph, o));
    }

    return result;
}

TF_SessionOptions* CreateSessionOptions(double gpu_memory_fraction, TF_Status* status) {
    // See https://github.com/Neargye/hello_tf_c_api/issues/21 for details.

    MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
    if (status == nullptr) {
        status = TF_NewStatus();
    } else {
        delete_status.dismiss();
    }

    auto options = TF_NewSessionOptions();

    // The following is an equivalent of setting this in Python:
    // config = tf.ConfigProto( allow_soft_placement = True )
    // config.gpu_options.allow_growth = True
    // config.gpu_options.per_process_gpu_memory_fraction = percentage
    // Create a byte-array for the serialized ProtoConfig, set the mandatory bytes (first three and last four)
    std::array<std::uint8_t, 15> config = {{0x32, 0xb, 0x9, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x20, 0x1, 0x38, 0x1}};

    // Convert the desired percentage into a byte-array.
    auto bytes = reinterpret_cast<std::uint8_t*>(&gpu_memory_fraction);

    // Put it to the config byte-array, from 3 to 10:
    for (std::size_t i = 0; i < sizeof(gpu_memory_fraction); ++i) {
        config[i + 3] = bytes[i];
    }

    TF_SetConfig(options, config.data(), config.size(), status);

    if (TF_GetCode(status) != TF_OK) {
        DeleteSessionOptions(options);
        return nullptr;
    }

    return options;
}

TF_SessionOptions* CreateSessionOptions(std::uint8_t intra_op_parallelism_threads, std::uint8_t inter_op_parallelism_threads, TF_Status* status) {
    // See https://github.com/tensorflow/tensorflow/issues/13853 for details.

    MAKE_SCOPE_EXIT(delete_status){ TF_DeleteStatus(status); };
    if (status == nullptr) {
        status = TF_NewStatus();
    } else {
        delete_status.dismiss();
    }

    auto options = TF_NewSessionOptions();
    std::array<std::uint8_t, 4> config = {{0x10, intra_op_parallelism_threads, 0x28, inter_op_parallelism_threads}};
    TF_SetConfig(options, config.data(), config.size(), status);

    if (TF_GetCode(status) != TF_OK) {
        DeleteSessionOptions(options);
        return nullptr;
    }

    return options;
}

void DeleteSessionOptions(TF_SessionOptions* options) {
    if (options != nullptr) {
        TF_DeleteSessionOptions(options);
    }
}

const char* DataTypeToString(TF_DataType data_type) {
    switch (data_type) {
    case TF_FLOAT:
        return "TF_FLOAT";
    case TF_DOUBLE:
        return "TF_DOUBLE";
    case TF_INT32:
        return "TF_INT32";
    case TF_UINT8:
        return "TF_UINT8";
    case TF_INT16:
        return "TF_INT16";
    case TF_INT8:
        return "TF_INT8";
    case TF_STRING:
        return "TF_STRING";
    case TF_COMPLEX64:
        return "TF_COMPLEX64";
    case TF_INT64:
        return "TF_INT64";
    case TF_BOOL:
        return "TF_BOOL";
    case TF_QINT8:
        return "TF_QINT8";
    case TF_QUINT8:
        return "TF_QUINT8";
    case TF_QINT32:
        return "TF_QINT32";
    case TF_BFLOAT16:
        return "TF_BFLOAT16";
    case TF_QINT16:
        return "TF_QINT16";
    case TF_QUINT16:
        return "TF_QUINT16";
    case TF_UINT16:
        return "TF_UINT16";
    case TF_COMPLEX128:
        return "TF_COMPLEX128";
    case TF_HALF:
        return "TF_HALF";
    case TF_RESOURCE:
        return "TF_RESOURCE";
    case TF_VARIANT:
        return "TF_VARIANT";
    case TF_UINT32:
        return "TF_UINT32";
    case TF_UINT64:
        return "TF_UINT64";
    default:
        return "Unknown";
    }
}

const char* CodeToString(TF_Code code) {
    switch (code) {
    case TF_OK:
        return "TF_OK";
    case TF_CANCELLED:
        return "TF_CANCELLED";
    case TF_UNKNOWN:
        return "TF_UNKNOWN";
    case TF_INVALID_ARGUMENT:
        return "TF_INVALID_ARGUMENT";
    case TF_DEADLINE_EXCEEDED:
        return "TF_DEADLINE_EXCEEDED";
    case TF_NOT_FOUND:
        return "TF_NOT_FOUND";
    case TF_ALREADY_EXISTS:
        return "TF_ALREADY_EXISTS";
    case TF_PERMISSION_DENIED:
        return "TF_PERMISSION_DENIED";
    case TF_UNAUTHENTICATED:
        return "TF_UNAUTHENTICATED";
    case TF_RESOURCE_EXHAUSTED:
        return "TF_RESOURCE_EXHAUSTED";
    case TF_FAILED_PRECONDITION:
        return "TF_FAILED_PRECONDITION";
    case TF_ABORTED:
        return "TF_ABORTED";
    case TF_OUT_OF_RANGE:
        return "TF_OUT_OF_RANGE";
    case TF_UNIMPLEMENTED:
        return "TF_UNIMPLEMENTED";
    case TF_INTERNAL:
        return "TF_INTERNAL";
    case TF_UNAVAILABLE:
        return "TF_UNAVAILABLE";
    case TF_DATA_LOSS:
        return "TF_DATA_LOSS";
    default:
        return "Unknown";
    }
}
}
/* NAMESPACE TensorFlow Utility ENDS */

/* NAMESPACE Classification USING TENSORFLOW CAPI */
namespace Classification
{

QDateTime timestamp_class;

CAPITfClassification::CAPITfClassification()
{

}

QString CAPITfClassification::ImageName() const
{
    return _ImageName;
}

void CAPITfClassification::setImageName(const QString &ImageName)
{
    _ImageName = ImageName;
}

std::string CAPITfClassification::ModelName() const
{
    return _ModelName;
}

void CAPITfClassification::setModelName(const std::string &ModelName)
{
    _ModelName = ModelName;
}

std::string CAPITfClassification::InputLayer() const
{
    return _InputLayer;
}

void CAPITfClassification::setInputLayer(const std::string &InputLayer)
{
    _InputLayer = InputLayer;
}

std::string CAPITfClassification::OutputLayer() const
{
    return _OutputLayer;
}

void CAPITfClassification::setOutputLayer(const std::string &OutputLayer)
{
    _OutputLayer = OutputLayer;
}

int CAPITfClassification::ImageWidth() const
{
    return _ImageWidth;
}

void CAPITfClassification::setImageWidth(int ImageWidth)
{
    _ImageWidth = ImageWidth;
}

int CAPITfClassification::ImageHeight() const
{
    return _ImageHeight;
}

void CAPITfClassification::setImageHeight(int ImageHeight)
{
    _ImageHeight = ImageHeight;
}

int CAPITfClassification::Channel() const
{
    return _Channel;
}

void CAPITfClassification::setChannel(int Channel)
{
    _Channel = Channel;
}

std::vector<std::int64_t> CAPITfClassification::Input_dims() const
{
    return _Input_dims;
}

void CAPITfClassification::setInput_dims(const std::vector<std::int64_t> &Input_dims)
{
    _Input_dims = Input_dims;
}


void CAPITfClassification::DeleteTensorVars()
{
    TF_DeleteSession(sess, status);
    TF_DeleteSessionOptions(options);
    TF_DeleteStatus(status);
    TF_DeleteGraph(graphptr);
}

void CAPITfClassification::Init(std::string ModelName,
                                 std::string InputLayer,
                                 std::string OutputLayer,
                                 int ImageWidth,
                                 int ImageHeight,
                                 int Channel
                                 )
{
    _ModelName = ModelName;
    _ImageHeight = ImageHeight;
    _ImageWidth = ImageWidth;
    _Channel = Channel;
    _InputLayer = InputLayer;
    _OutputLayer = OutputLayer;
    _Input_dims = {1,_ImageWidth,_ImageHeight,_Channel};

    std::cout<<"_ModelName:"<<_ModelName<<std::endl;
    std::cout<<"_ImageHeight:"<<_ImageHeight<<std::endl;
    std::cout<<"_ImageWidth:"<<_ImageWidth<<std::endl;
    std::cout<<"_Channel:"<<_Channel<<std::endl;
    std::cout<<"_InputLayer:"<<_InputLayer<<std::endl;
    std::cout<<"_OutputLayer:"<<_OutputLayer<<std::endl;
    std::cout<<"_Input_dims:"<<_Input_dims[0]<<","
            <<_Input_dims[1]<<","
           <<_Input_dims[2]<<","
          <<_Input_dims[3]<<","
         <<std::endl;

    //Load Graph
    graphptr = LoadGraph(_ModelName.c_str());
    // Auto-delete on scope exit.
    if (graphptr == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        return;
    }
    std::cout << "Graph loaded" << std::endl;

    //Create Graph operation By name, give input layer name
    input_op = TF_Output{TF_GraphOperationByName(graphptr, _InputLayer.c_str()), 0};
    if (input_op.oper == nullptr) {
        std::cout << "Can't init input_op" << std::endl;
        return;
    }
    std::cout << "INOP loaded" << std::endl;

    //Create Output operation, give output layer name
    output_op = TF_Output{TF_GraphOperationByName(graphptr, _OutputLayer.c_str()), 0};
    if (output_op.oper == nullptr) {
        std::cout << "Can't init out_op" << std::endl;
        return;
    }
    std::cout << "OPOP loaded" << std::endl;

    status = TF_NewStatus();
    options = TF_NewSessionOptions();
    sess = TF_NewSession(graphptr, options, status);

}

std::vector<float> CAPITfClassification::ConvertQImageToFloat(QImage PrepImage)
{
    QImage scaledImage = PrepImage.scaled(_ImageWidth,
                                          _ImageHeight,
                                          Qt::IgnoreAspectRatio,
                                          Qt::FastTransformation);

    scaledImage=scaledImage.convertToFormat(QImage::Format_RGB888,Qt::AutoColor);

    //Image edit
    int64_t imglen=_ImageWidth * _ImageHeight *3;
    std::cout<<imglen<<std::endl;

    float* inputs = new float[imglen];
    unsigned char* uimg = scaledImage.bits();

    for(int i=0;i<_ImageHeight;i++)
    {
        for(int j=0;j<_ImageWidth;j++)
        {
            inputs[i*_ImageWidth*3+j*3]=float(uimg[i*_ImageWidth*3+j*3]);
            inputs[i*_ImageWidth*3+j*3+1]=float(uimg[i*_ImageWidth*3+j*3+1]);
            inputs[i*_ImageWidth*3+j*3+2]=float(uimg[i*_ImageWidth*3+j*3+2]);
        }
    }

    std::vector<float> val; // Float Vector For Image

    //Convert to Float Vector
    for(int i=0;i<imglen;i++)
    {
        val.push_back(inputs[i]);
    }

    int x= val.size();
    std::cout<<"size:"<<x<<std::endl; //returns size of bytearray derived from image
    delete [] inputs;
    return val;

}

//std::vector<float> TFClassification::ConvertCVMatToFloat(cv::Mat PrepImage)
//{
//    std::vector<float> val;
//    return val;
//}

float* CAPITfClassification::RunInferClassification(QImage InferImage)
{
    //Timer For Benchmark
    QTime myTimer;
    myTimer.start();

    std::cout<<timestamp_class.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Classification Image to float " << std::endl;
    //Create Input Tensor using datatype dimension and image in float array/vector
    auto input_tensor = CreateTensor(TF_FLOAT, _Input_dims, ConvertQImageToFloat(InferImage));
    SCOPE_EXIT{ DeleteTensor(input_tensor); }; // Auto-delete on scope exit.
    std::cout<<timestamp_class.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Classification Input tensor created " << std::endl;

    //Create Empty Output tensor
    TF_Tensor* output_tensor =nullptr;
    SCOPE_EXIT{ DeleteTensor(output_tensor); }; // Auto-delete on scope exit.
    std::cout<<timestamp_class.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Classification Output tensor created " << std::endl;

    //Declare Session vars
    //auto status = TF_NewStatus();
    //SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
    //std::cout<<"tensor status created"<<std::endl;
    //auto options = TF_NewSessionOptions();
    //std::cout<<"tensor option created"<<std::endl;
    //auto sess = TF_NewSession(graphptr, options, status);

    //auto sess = TF_NewSession(graphptr, options, status);
    //std::cout<<"tensor sess created"<<std::endl;
    //TF_DeleteSessionOptions(options);
    //TF_DeleteStatus(status);

    std::cout<<"tensor var created"<<std::endl;

    //    if (TF_GetCode(status) != TF_OK) {
    //        return _MaskMap;
    //    }

    //    PrintTensorInfo(graphptr, _InputLayer, status);
    //    std::cout<<"tt"<< std::endl;
    //    PrintTensorInfo(graphptr, _OutputLayer, status);
    //    std::cout <<"tt"<< std::endl;



    //Running new session
    std::cout<<timestamp_class.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Classification Tensor session start " << std::endl;

    TF_SessionRun(sess,
                  NULL, // Run options.
                  &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                  &output_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                  NULL, 0, // Target operations, number of targets.
                  NULL, // Run metadata.
                  status // Output status.
                  );

    if (TF_GetCode(status) != TF_OK) {
        std::cout << "Error run session"<<std::endl;
        return NULL;
    }

    //TF_CloseSession(sess, status);

    std::cout<<timestamp_class.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Classification Tensor session end " << std::endl;

    if (TF_GetCode(status) != TF_OK) {
        std::cout << "Error close session"<<std::endl;
        return NULL;
    }

    //TF_DeleteSession(sess, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << "Error delete session"<<std::endl;
        return NULL;
    }

    //Converting to float pointer array, getting float pointer
    auto data = static_cast<float*>(TF_TensorData(output_tensor));
    std::cout<<timestamp_class.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Classification Process time : "<<myTimer.elapsed()<<std::endl;

    return data;

}
}
/* NAMESPACE Classification ENDS */

/* NAMESPACE Segmentation USING TENSORFLOW CAPI */
namespace Segmentation
{

QDateTime timestamp_tfs;

CAPITfSegmentation::CAPITfSegmentation()
{

}
QString CAPITfSegmentation::ImageName() const
{
    return _ImageName;
}

void CAPITfSegmentation::setImageName(const QString &ImageName)
{
    _ImageName = ImageName;
}

std::string CAPITfSegmentation::ModelName() const
{
    return _ModelName;
}

void CAPITfSegmentation::setModelName(const std::string &ModelName)
{
    _ModelName = ModelName;
}

std::string CAPITfSegmentation::InputLayer() const
{
    return _InputLayer;
}

void CAPITfSegmentation::setInputLayer(const std::string &InputLayer)
{
    _InputLayer = InputLayer;
}

std::string CAPITfSegmentation::OutputLayer() const
{
    return _OutputLayer;
}

void  CAPITfSegmentation::setOutputLayer(const std::string &OutputLayer)
{
    _OutputLayer = OutputLayer;
}

int CAPITfSegmentation::ImageWidth() const
{
    return _ImageWidth;
}

void CAPITfSegmentation::setImageWidth(int ImageWidth)
{
    _ImageWidth = ImageWidth;
}

int CAPITfSegmentation::ImageHeight() const
{
    return _ImageHeight;
}

void CAPITfSegmentation::setImageHeight(int ImageHeight)
{
    _ImageHeight = ImageHeight;
}

int CAPITfSegmentation::Channel() const
{
    return _Channel;
}

void CAPITfSegmentation::setChannel(int Channel)
{
    _Channel = Channel;
}

std::vector<std::int64_t> CAPITfSegmentation::Input_dims() const
{
    return _Input_dims;
}

void CAPITfSegmentation::setInput_dims(const std::vector<std::int64_t> &Input_dims)
{
    _Input_dims = Input_dims;
}


void CAPITfSegmentation::DeleteTensorVars()
{
    TF_DeleteSession(sess, status);
    TF_DeleteSessionOptions(options);
    TF_DeleteStatus(status);
    TF_DeleteGraph(graphptr);
}

void CAPITfSegmentation::Init(std::string ModelName,
                               std::string InputLayer,
                               std::string OutputLayer,
                               int ImageWidth,
                               int ImageHeight,
                               int Channel
                               )
{
    _ModelName = ModelName;
    _ImageHeight = ImageHeight;
    _ImageWidth = ImageWidth;
    _Channel = Channel;
    _InputLayer = InputLayer;
    _OutputLayer = OutputLayer;
    _Input_dims = {1,_ImageWidth,_ImageHeight,3};

    std::cout<<"_ModelName:"<<_ModelName<<std::endl;
    std::cout<<"_ImageHeight:"<<_ImageHeight<<std::endl;
    std::cout<<"_ImageWidth:"<<_ImageWidth<<std::endl;
    std::cout<<"_Channel:"<<_Channel<<std::endl;
    std::cout<<"_InputLayer:"<<_InputLayer<<std::endl;
    std::cout<<"_OutputLayer:"<<_OutputLayer<<std::endl;
    std::cout<<"_Input_dims:"<<_Input_dims[0]<<","
            <<_Input_dims[1]<<","
           <<_Input_dims[2]<<","
          <<_Input_dims[3]<<","
         <<std::endl;

    //Load Graph
    graphptr = LoadGraph(_ModelName.c_str());
    //auto graph = LoadGraph(_ModelName.toUtf8().constData());
    // Auto-delete on scope exit.
    if (graphptr == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        return;
    }
    std::cout << "Graph loaded" << std::endl;

    //Create Graph operation By name, give input layer name
    input_op = TF_Output{TF_GraphOperationByName(graphptr, _InputLayer.c_str()), 0};
    //auto input_op = TF_Output{TF_GraphOperationByName(graphptr, _InputLayer), 0};
    if (input_op.oper == nullptr) {
        std::cout << "Can't init input_op" << std::endl;
        return;
    }
    std::cout << "INOP loaded" << std::endl;

    //Create Output operation, give output layer name
    output_op = TF_Output{TF_GraphOperationByName(graphptr, _OutputLayer.c_str()), 0};
    // auto out_op = TF_Output{TF_GraphOperationByName(graphptr, _OutputLayer), 0};
    if (output_op.oper == nullptr) {
        std::cout << "Can't init out_op" << std::endl;
        return;
    }
    std::cout << "OPOP loaded" << std::endl;

    status = TF_NewStatus();
    //SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
    options = TF_NewSessionOptions();
    sess = TF_NewSession(graphptr, options, status);

}

std::vector<float> CAPITfSegmentation::ConvertQImageToFloat(QImage PrepImage)
{
    std::vector<float> val; // Float Vector For Image

    try
    {
        QImage scaledImage = PrepImage.scaled(_ImageWidth,
                                              _ImageHeight,
                                              Qt::IgnoreAspectRatio,
                                              Qt::FastTransformation);

        scaledImage=scaledImage.convertToFormat(QImage::Format_RGB888,Qt::AutoColor);

        //Image edit
        int64_t imglen=_ImageWidth * _ImageHeight *3;
        std::cout<<imglen<<std::endl;

        //float* inputs = new float[imglen];
        unsigned char* uimg = scaledImage.bits();
        std::vector<float> val{}; // Float Vector For Image

        for(int i=0;i<_ImageHeight;i++)
        {
            for(int j=0;j<_ImageWidth;j++)
            {
                val.push_back(float(uimg[i*_ImageWidth*3+j*3]));
                val.push_back(float(uimg[i*_ImageWidth*3+j*3+1]));
                val.push_back(float(uimg[i*_ImageWidth*3+j*3+2]));
            }
        }

        int x= val.size();
        std::cout<<"size:"<<x<<std::endl; //returns size of bytearray derived from image

        return val;

    }
    catch(QException exp)
    {

    }


    //  return val;



}

//std::vector<float> CAPITfSegmentation::ConvertCVMatToFloat(cv::Mat PrepImage)
//{
//    std::vector<float> val;
//    return val;
//}

QMap<QString, QImage> CAPITfSegmentation::RunInferSegmentation(QImage InferImage)
{
    //Timer For Benchmark
    QTime myTimer;
    myTimer.start();

    QMap<QString,QImage> ret;

    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Create Input Tensor " << std::endl;
    //Create Input Tensor using datatype dimension and image in float array/vector
    auto input_tensor = CreateTensor(TF_FLOAT, _Input_dims, ConvertQImageToFloat(InferImage));
    SCOPE_EXIT{ DeleteTensor(input_tensor); }; // Auto-delete on scope exit.
    std::cout<<"input tensor created"<<std::endl;
    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Input tensor created " << std::endl;

    //Create Empty Output tensor
    TF_Tensor* output_tensor =nullptr;
    SCOPE_EXIT{ DeleteTensor(output_tensor); }; // Auto-delete on scope exit.
    //std::cout<<"output tensor created"<<std::endl;
    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Output tensor created " << std::endl;
    //Declare Session vars
    //auto status = TF_NewStatus();
    //SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
    //std::cout<<"tensor status created"<<std::endl;
    //auto options = TF_NewSessionOptions();
    //std::cout<<"tensor option created"<<std::endl;
    //auto sess = TF_NewSession(graphptr, options, status);

    //auto sess = TF_NewSession(graphptr, options, status);
    //std::cout<<"tensor sess created"<<std::endl;
    //TF_DeleteSessionOptions(options);


    //std::cout<<"tensor var created"<<std::endl;

    //    if (TF_GetCode(status) != TF_OK) {
    //        return _MaskMap;
    //    }

    //    PrintTensorInfo(graphptr, _InputLayer, status);
    //    std::cout<<"tt"<< std::endl;
    //    PrintTensorInfo(graphptr, _OutputLayer, status);
    //    std::cout <<"tt"<< std::endl;



    //Running new session
    //std::cout<<"tensor session reached"<<std::endl;
    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Run Tensor Session " << std::endl;
    TF_SessionRun(sess,
                  NULL, // Run options.
                  &input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
                  &output_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
                  NULL, 0, // Target operations, number of targets.
                  NULL, // Run metadata.
                  status // Output status.
                  );

    if (TF_GetCode(status) != TF_OK) {
        std::cout << "Error run session"<<std::endl;
        return ret;
    }

    //TF_CloseSession(sess, status);

    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< "  Tensor Session ended " << std::endl;

    if (TF_GetCode(status) != TF_OK) {
        std::cout << "Error close session"<<std::endl;
        return ret;
    }

    //TF_DeleteSession(sess, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cout << "Error delete session"<<std::endl;
        return ret;
    }

    //Converting to float pointer array, getting float pointer
    auto data = static_cast<float*>(TF_TensorData(output_tensor));
    ret.clear();
    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< " Convert Tensor output to QImage " << std::endl;
    ret=ConvertFloatToQImage(data);

    std::cout<<timestamp_tfs.currentDateTime().toString("yyyy-MM-dd-hh:mm:ss:zzz").toStdString()<< " Total Segmentation time : " <<myTimer.elapsed()<<" ms "<< std::endl;
    std::cout<<"segmentation time:"<<myTimer.elapsed()<<std::endl;

    return ret;

}

//QMap<QString, QImage> CAPITfSegmentation::RunInferSegmentation(cv::Mat InferImage)
//{
//    QMap<QString, QImage> ret;
//    return ret;
//}

QMap<QString,QImage> CAPITfSegmentation::ConvertFloatToQImage(float* outputTensor)
{
    QImage mask_1(_ImageWidth,_ImageHeight,QImage::Format_RGB888);
    QImage mask_2(_ImageWidth,_ImageHeight,QImage::Format_RGB888);
    QImage mask_3(_ImageWidth,_ImageHeight,QImage::Format_RGB888);
    QMap<QString,QImage> maskMap;
    try
    {
        int pixel_mask_1;
        int pixel_mask_2;
        int pixel_mask_3;
        if(_Channel==1)
        {
            for(int i=0;i<_ImageHeight;i++)
            {
                for(int j=0;j<_ImageWidth;j++)
                {
                    pixel_mask_1 = outputTensor[i*_ImageWidth+j]*255;
                    pixel_mask_2 = outputTensor[(i*_ImageHeight+j)+1]*255;
                    pixel_mask_3 = outputTensor[(i*_ImageHeight+j)+2]*255;

                    if(pixel_mask_1>0.5)
                    {
                        mask_1.setPixel(j,i,qRgb(pixel_mask_1,pixel_mask_1,pixel_mask_1));
                        mask_2.setPixel(j,i,qRgb(pixel_mask_2,pixel_mask_2,pixel_mask_2));
                        mask_3.setPixel(j,i,qRgb(pixel_mask_3,pixel_mask_3,pixel_mask_3));
                    }
                    else
                    {
                        mask_1.setPixel(j,i,qRgb(255,0,0));
                        mask_2.setPixel(j,i,qRgb(255,0,0));
                        mask_3.setPixel(j,i,qRgb(255,0,0));
                    }
                }
            }

            maskMap.insert("Background",mask_1);
            maskMap.insert("Defect",mask_2);
            maskMap.insert("Body",mask_3);
        }
        else
        {

            for(int i=0;i<_ImageHeight;i++)
            {
                for(int j=0;j<_ImageWidth;j++)
                {
                    pixel_mask_1= outputTensor[i*_ImageWidth*3+j*3]*255;
                    pixel_mask_2 = outputTensor[(i*_ImageHeight*3+j*3)+1]*255;
                    pixel_mask_3 = outputTensor[(i*_ImageHeight*3+j*3)+2]*255;

                    mask_1.setPixel(j,i,qRgb(pixel_mask_1,pixel_mask_1,pixel_mask_1));
                    mask_2.setPixel(j,i,qRgb(pixel_mask_2,pixel_mask_2,pixel_mask_2));
                    mask_3.setPixel(j,i,qRgb(pixel_mask_3,pixel_mask_3,pixel_mask_3));
                }
            }
            maskMap.insert("Background",mask_1);
            maskMap.insert("Defect",mask_2);
            maskMap.insert("Body",mask_3);
        }
    }
    catch(QException exp)
    {
        std::cout << "Exception in 'ConvertFloatToQImage' is "<<exp.what() << std::endl;
    }

    return maskMap;
}
}
/* NAMESPACE Segmentation ENDS */

/* NAMESPACE Segmentation_PostProcessing USING OPENCV API */
namespace Segmentation_PostProcessing
{
cv::Mat CAPISegmentation::QImageToCvMat( QImage image)
{
//    cv::Mat  mat( inImage.height(), inImage.width(),
//                              CV_8UC4,
//                              const_cast<uchar*>(inImage.bits()),
//                              static_cast<size_t>(inImage.bytesPerLine())
//                              );
//                cv::Mat  matImage;
//                cv::cvtColor( mat, matImage, cv::COLOR_BGRA2BGR );   // drop the all-white alpha channel

//                return matImage;
    cv::Mat out{};
          switch(image.format())
          {
          case QImage::Format_RGB32:
              {
                 cv::Mat imageMat{};
                 Mat res_img(image.height(),image.width(),CV_8UC4,(void *)image.constBits(),image.bytesPerLine());
                 cv::cvtColor(res_img, imageMat,COLOR_RGBA2RGB );
                 imageMat.copyTo(out);
                 break ;
              }
          case QImage::Format_RGB888:
              {
                 cv::Mat imageMat{};
                 Mat res_img(image.height(),image.width(),CV_8UC3,(void *)image.constBits(),image.bytesPerLine());
                 cvtColor(res_img, imageMat,COLOR_RGBA2RGB );
                 imageMat.copyTo(out);
                 break ;
              }
          case QImage::Format_Invalid:
              {
                 Mat empty;
                 empty.copyTo(out);
                 break ;
              }
          default:
          {
              cv::Mat imageMat{};
              cv::Mat res_img(image.height(), image.width(),CV_8UC4,
                                    const_cast<uchar*>(image.bits()), image.bytesPerLine());
              cv::cvtColor(res_img, imageMat, COLOR_RGBA2RGB );
                 imageMat.copyTo(out);
                 break;
          };
          }
          return  out;
}

 QMap<QString,vector<vector<Point>>> CAPISegmentation::postProcess(QMap<QString,QImage>imageMAp,int height,int width,double param_threshold)
 {
     //Function call to convert QImage to mat image
     //int tempID=0;
     //QList<vector<vector<Point>>> contour_xy;
     QMap<QString,vector<vector<Point>>>contour_xy;
     vector<vector<Point>>contour_points;

     std::vector<cv::Mat> mVector;
     for(QImage im : imageMAp)
     {
         cv::Mat cv_Image= QImageToCvMat(im);
         mVector.push_back(cv_Image);

     }

     //Function call to find contours
     for(cv::Mat mat : mVector)
     {
        // switch (Id)
         {
        // case 0:
             //contour_points= find_Contours(mat,param_threshold);
             //contour_xy.insert("Background",contour_points);
//             namedWindow("Backgroubd_raw", WINDOW_NORMAL);
//             imshow("Background_raw",mat);
             //break;

         //case 1:
            // contour_points= find_Contours(mat,param_threshold);
             //contour_xy.insert("Body",contour_points);
//             namedWindow("Body_raw", WINDOW_NORMAL);
//             imshow("Body_raw",mat);
             //break;

         //case 2:
             contour_points= find_Contours(mat,param_threshold);
             contour_xy.insert("Defect",contour_points);
             //contour_xy.push_back(contour_points);
//             namedWindow("Defect_raw", WINDOW_NORMAL);
//             imshow("Defect_raw",mat);

             //break;

         }
         //Id++;

//         contour_points= find_Contours(mat,param_threshold);
//         contour_xy.insert(to_string(Id),contour_points);
//         Id++;
     }


     return contour_xy;

     contour_xy.clear();
     contour_points.clear();
     mVector.clear();
     imageMAp.clear();

 }



 vector<vector<Point>> CAPISegmentation::find_Contours(cv::Mat im,double threshold_param)
 {
     cv::Mat im_gray;
     cv::Mat im_threshold;
     cv::cvtColor(im,im_gray,cv::COLOR_RGB2GRAY);
     cv::threshold(im_gray,im_threshold,threshold_param,255,cv::THRESH_BINARY);
     contourOutput =im_threshold.clone(); //mat.clone();
     findContours(contourOutput,contours,RETR_CCOMP,CHAIN_APPROX_NONE);
     Mat im_contour(im.size(), CV_8UC3);

//     for (size_t idx = 0; idx < contours.size(); idx++)
//     {
//         cv::drawContours(im_contour,contours,idx,Scalar(128,255,255),2);//(im_contour, contours, idx, colors[idx % 3]);
//     }
//     cv::resizeWindow("myContour",800,800);
//     imshow("myContour",im_contour);

     std::vector<std::vector<cv::Point>> pt;
     for(int a=0; a<contours.size(); a++)
     {
         double area= contourArea(contours[a],false);
         if ( area > 1000 )
         {
            pt.push_back(contours.at(a));
         }
     }
     return pt;

     //polylines(im,pt,false,Scalar(255,0,0),2);


 }

CAPISegmentation::CAPISegmentation()
{

}
}
/* NAMESPACE Segmentation_PostProcessing ENDS */

/* NAMESPACE ObjectDetection USING TENSORFLOW CAPI */
namespace ObjectDetection
{

}
/* NAMESPACE ObjectDetection USING ENDS */

}
