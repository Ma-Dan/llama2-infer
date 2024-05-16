#include "swish.h"
#include "layer_register.h"
#include "utils.h"
#include <cmath>

Swish::Swish()
{
}

Swish::~Swish()
{
}

int Swish::load_model(const vector<string> &params, FILE* fp)
{
    return 0;
}

void Swish::forward(vector<Tensor*> &input, vector<Tensor*> &output)
{
    Tensor* result;
    if(output[0] == nullptr)
    {
        result = new Tensor();
    }
    else
    {
        result = output[0];
    }

    vector<float>* input0Data = input[0]->get_data();
    vector<int> input0Shape = input[0]->get_shape();

    result->set_shape(input0Shape);
    vector<float>* outputData = result->get_data();

    //#pragma omp parallel for
    for(int i=0; i<input0Data->size(); i++)
    {
        outputData->data()[i] = input0Data->data()[i] * (1.0f / (1.0f + expf(-input0Data->data()[i])));
    }

    output[0] = result;
}

int Swish::CreateInstance(Layer* &layer)
{
    layer = new Swish();
    return 0;
}

LayerRegistererWrapper swishCreateInstance("Swish_t", Swish::CreateInstance);