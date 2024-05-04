#include "softmax.h"
#include "layer_register.h"
#include "utils.h"
#include <cmath>

Softmax::Softmax()
{
}

Softmax::~Softmax()
{
}

int Softmax::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> dim_param = split(params[6], "=");
    _dim = atoi(dim_param[1].c_str());

    return 0;
}

void Softmax::forward(vector<Tensor*> &input, vector<Tensor*> &output)
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

    if(_dim == 0)
    {
        //TODO:处理维度更多情况
        vector<int> inputShape = input[0]->get_shape();
        result->set_shape(inputShape);
        vector<float>* outputData = result->get_data();

        vector<double> calcData(outputData->size(), 0);

        for(int i=0; i<outputData->size(); i++)
        {
            calcData.data()[i] = exp2l(inputData->data()[i]);
        }

        for(int i=0; i<inputShape[1]; i++)
        {
            double sum = 0.0f;
            for(int j=0; j<inputShape[0]; j++)
            {
                sum += calcData.data()[j*inputShape[1]+i];
            }
            for(int j=0; j<inputShape[0]; j++)
            {
                outputData->data()[j*inputShape[1]+i] = calcData.data()[j*inputShape[1]+i] /= sum;
            }
        }

        output[0] = result;
    }
}

int Softmax::CreateInstance(Layer* &layer)
{
    layer = new Softmax();
    return 0;
}

LayerRegistererWrapper softmaxCreateInstance("Softmax_t", Softmax::CreateInstance);