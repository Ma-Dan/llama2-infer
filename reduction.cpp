#include "reduction.h"
#include "layer_register.h"
#include "utils.h"

Reduction::Reduction()
{
    _reduce_type = -1;
}

Reduction::~Reduction()
{
}

int Reduction::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> reduce_param = split(params[6], "=");
    if(reduce_param[1] == "Mean")
    {
        _reduce_type = 0;
    }

    vector<string> dim_param = split(params[7], "=");
    _dim = atoi(dim_param[1].c_str());

    return 0;
}

void Reduction::forward(vector<Tensor*> &input, vector<Tensor*> &output)
{
    if(_reduce_type==0 && _dim == -1)
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

        vector<int> shape;
        shape.push_back(1);

        result->set_shape(shape);

        float sum = 0.0f;
        vector<float>* inputData = input[0]->get_data();
        for(int i=0; i<inputData->size(); i++)
        {
            sum += inputData->data()[i];
        }

        vector<float>* outputData = result->get_data();
        outputData->data()[0] = sum / inputData->size();

        output[0] = result;
    }
}

int Reduction::CreateInstance(Layer* &layer)
{
    layer = new Reduction();
    return 0;
}

LayerRegistererWrapper reductionCreateInstance("Reduction_t", Reduction::CreateInstance);