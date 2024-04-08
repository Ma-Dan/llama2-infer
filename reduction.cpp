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
    Tensor* result;
    if(output[0] == nullptr)
    {
        result = new Tensor();
    }
    else
    {
        result = output[0];
    }

    vector<float>* inputData = input[0]->get_data();
    vector<float>* outputData = result->get_data();

    if(_reduce_type == 0)
    {
        //TODO:处理维度更多情况
        vector<int> inputShape = input[0]->get_shape();
        vector<int> outputShape;

        int blocks = 1;
        for(int i=0; i<_dim; i++)
        {
            outputShape.push_back(inputShape[i]);
            blocks *= inputShape[i];
        }

        int stride = 1;
        for(int i=_dim; i<inputShape.size(); i++)
        {
            stride *= inputShape[i];
        }

        result->set_shape(outputShape);

        for(int i=0; i<blocks; i++)
        {
            float sum = 0.0f;
            for(int j=0; j<stride; j++)
            {
                sum += inputData->data()[i*stride+j];
            }
            outputData->data()[i] = sum / stride;
        }

        output[0] = result;
    }
}

int Reduction::CreateInstance(Layer* &layer)
{
    layer = new Reduction();
    return 0;
}

LayerRegistererWrapper reductionCreateInstance("Reduction_t", Reduction::CreateInstance);