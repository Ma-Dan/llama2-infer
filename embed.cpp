#include "embed.h"
#include "layer_register.h"
#include "utils.h"

Embed::Embed()
{
    _weight = new Tensor();
}

Embed::~Embed()
{
    delete _weight;
}

int Embed::load_model(const vector<string> &params, FILE* fp)
{
    vector<int> shape;
    for(int i=0; i<2; i++)
    {
        vector<string> dim_size = split(params[7-i], "=");
        shape.push_back(atoi(dim_size[1].c_str()));
    }

    _weight->set_shape(shape);

    _weight->load_data(fp);

    return 0;
}

void Embed::forward(vector<Tensor*> &input, vector<Tensor*> &output)
{
    int index = (int)(input[0]->get_data()[0]);

    vector<int> shape;
    shape.push_back(_weight->get_shape()[1]);

    if(output[0] == nullptr)
    {
        Tensor* result = new Tensor();
        result->set_shape(shape);

        vector<float> data;
        data.resize(shape[0]);

        memcpy(data.data(), &_weight->get_data()[index*shape[0]], sizeof(float)*shape[0]);
        result->set_data(data);

        output[0] = result;
    }
    else
    {
        output[0]->set_shape(shape);
        memcpy(output[0]->get_data().data(), &_weight->get_data()[index*_weight->get_shape()[0]], _weight->get_shape()[1]*sizeof(float));
    }
}

int Embed::CreateInstance(Layer* &layer)
{
    layer = new Embed();
    return 0;
}

LayerRegistererWrapper embedCreateInstance("Embed_t", Embed::CreateInstance);