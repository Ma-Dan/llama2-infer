#include "embed.h"
#include "layer_register.h"
#include "utils.h"

Embed::Embed()
{
    _weight = new Tensor();
}

Embed::~Embed()
{
    SAFE_DELETE(_weight);
}

int Embed::load_model(const vector<string> &params, FILE* fp)
{
    vector<int> shape;
    for(int i=0; i<2; i++)
    {
        vector<string> dim_size = split(params[7-i], "=");
        shape.push_back(atoi(dim_size[1].c_str()));
    }

    vector<string> weight_offset_param = split(params[8], "=");
    long weight_offset = atol(weight_offset_param[1].c_str());

    _weight->set_shape(shape);

    _weight->load_data(fp, weight_offset);

    return 0;
}

void Embed::forward(vector<Tensor*> &input, vector<Tensor*> &output)
{
    //TODO:更多的shape处理
    //暂时按照 长度 一个维度处理
    int dim0 = input[0]->get_shape()[0];

    vector<int> shape;
    shape.push_back(dim0);
    shape.push_back(_weight->get_shape()[1]);

    Tensor* result;
    if(output[0] == nullptr)
    {
        result = new Tensor();
    }
    else
    {
        result = output[0];
    }

    result->set_shape(shape);
    vector<float>* data = result->get_data();

    for(int i=0; i<dim0; i++)
    {
        int index = (int)((*(input[0]->get_data()))[i]);
        memcpy(&data->data()[i*shape[1]], &_weight->get_data()->data()[index*shape[1]], sizeof(float)*shape[1]);
    }

    output[0] = result;
}

int Embed::CreateInstance(Layer* &layer)
{
    layer = new Embed();
    return 0;
}

LayerRegistererWrapper embedCreateInstance("Embed_t", Embed::CreateInstance);