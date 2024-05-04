#include "memorydata.h"
#include "layer_register.h"
#include "utils.h"

MemoryData::MemoryData()
{
    _weight = new Tensor();
}

MemoryData::~MemoryData()
{
    SAFE_DELETE(_weight);
}

int MemoryData::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> shape_dim_param = split(params[5], "=");
    int shape_dim = atoi(shape_dim_param[1].c_str());

    vector<int> shape;
    for(int i=0; i<shape_dim; i++)
    {
        vector<string> dim_size_param = split(params[6+i], "=");
        shape.push_back(atoi(dim_size_param[1].c_str()));
    }

    vector<string> bias_param = split(params[8+shape_dim], "=");
    int bias = atoi(bias_param[1].c_str());
    _weight->set_bias(bias);

    _weight->set_shape(shape);

    vector<string> weight_offset_param = split(params[6+shape_dim], "=");
    long weight_offset = atol(weight_offset_param[1].c_str());

    vector<string> device_param = split(params[7+shape_dim], "=");
    if("GPU" == device_param[1])
    {
        _weight->set_device_type(Tensor_GPU);
    }

    _weight->load_data(fp, weight_offset*sizeof(float));

    return 0;
}

void MemoryData::forward(vector<Tensor*> &input, vector<Tensor*> &output)
{
    /*Tensor* result;

    if(output[0] == nullptr)
    {
        result = new Tensor();
    }
    else
    {
        result = output[0];
    }

    result->set_shape(_weight->get_shape());

    vector<float>* data = result->get_data();
    memcpy(data->data(), &_weight->get_data()->data()[0], sizeof(float)*_weight->get_data()->size());

    output[0] = result;*/
    //TODO:暂时改成这样避免数据复制
    output[0] = _weight;
}

int MemoryData::CreateInstance(Layer* &layer)
{
    layer = new MemoryData();
    return 0;
}

LayerRegistererWrapper memoryDataCreateInstance("MemoryData_t", MemoryData::CreateInstance);