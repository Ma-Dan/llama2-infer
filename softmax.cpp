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

        int i;
        #pragma omp parallel for private(i)
        for(i=0; i<inputShape[1]; i++)
        {
            vector<float> m(inputShape[0]+1);
            vector<float> d(inputShape[0]+1);
            m[0] = -1e10;
            d[0] = 0;
            for(int j=0; j<inputShape[0]; j++)
            {
                float x = inputData->data()[j*inputShape[1]+i];
                m[j+1] = max(m[j], x);
                d[j+1] = d[j]*expf(m[j]-m[j+1])+expf(x-m[j+1]);
            }
            for(int j=0; j<inputShape[0]; j++)
            {
                float x = inputData->data()[j*inputShape[1]+i];
                outputData->data()[j*inputShape[1]+i] = expf(x-m[inputShape[0]])/d[inputShape[0]];
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