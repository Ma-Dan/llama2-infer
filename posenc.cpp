#include "posenc.h"
#include "layer_register.h"
#include "utils.h"
#include "math.h"

Posenc::Posenc()
{
    _use_last = 0;
}

Posenc::~Posenc()
{
}

int Posenc::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> use_last_param = split(params[8], "=");
    _use_last = atoi(use_last_param[1].c_str());

    return 0;
}

void Posenc::forward(vector<Tensor*> &input, vector<Tensor*> &output)
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

    //TODO:判断更多尺寸情况
    vector<float>* inputData = input[0]->get_data();
    vector<int> outputShape = input[0]->get_shape();
    result->set_shape(outputShape);
    vector<float>* outputData = result->get_data();

    vector<float>* freqs_cos = input[1]->get_data();
    vector<float>* freqs_sin = input[2]->get_data();

    for(int i=0; i<outputShape[0]; i++)
    {
        int freqOffset;
        if(_use_last)
        {
            freqOffset = (input[1]->get_shape()[0]-1)*input[1]->get_shape()[1];
        }
        else
        {
            freqOffset = i*input[1]->get_shape()[1];
        }
        for(int j=0; j<outputShape[1]; j++)
        {
            int dataOffset = (i*outputShape[1] + j)*outputShape[2];
            for(int k=0; k<outputShape[2]; k++)
            {
                float sign = -1;
                if(k >= outputShape[2]/2) {
                    sign = 1;
                }
                outputData->data()[dataOffset+k] = inputData->data()[dataOffset+k] * freqs_cos->data()[freqOffset+k%(outputShape[2]/2)] + sign * inputData->data()[dataOffset+(k+outputShape[2]/2)%outputShape[2]] * freqs_sin->data()[freqOffset+k%(outputShape[2]/2)];
            }
        }
    }

    output[0] = result;
}

int Posenc::CreateInstance(Layer* &layer)
{
    layer = new Posenc();
    return 0;
}

LayerRegistererWrapper posencCreateInstance("Posenc_t", Posenc::CreateInstance);