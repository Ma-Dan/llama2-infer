#include "matmul.h"
#include "layer_register.h"
#include "utils.h"

Matmul::Matmul()
{
}

Matmul::~Matmul()
{
}

int Matmul::load_model(const vector<string> &params, FILE* fp)
{
    return 0;
}

void Matmul::forward(vector<Tensor*> &input, vector<Tensor*> &output)
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
    vector<float>* input1Data = input[1]->get_data();
    vector<int> input0Shape = input[0]->get_shape();
    vector<int> input1Shape = input[1]->get_shape();

    if(input0Shape.size() == 2)
    {
        if(input0Shape[1] != input1Shape[1])
        {
            return;
        }

        vector<int> outputShape;
        outputShape.push_back(input0Shape[0]);
        outputShape.push_back(input1Shape[0]);
        result->set_shape(outputShape);
        vector<float>* outputData = result->get_data();

        for(int i=0; i<input0Shape[0]; i++)
        {
            for(int j=0; j<input1Shape[0]; j++)
            {
                float sum = 0.0f;
                for(int k=0; k<input0Shape[1]; k++)
                {
                    sum += input0Data->data()[i*input0Shape[1]+k] * input1Data->data()[j*input1Shape[1]+k];
                }
                outputData->data()[i*input1Shape[0]+j] = sum;
            }
        }
    }
    else if(input0Shape.size() == 3)
    {
        if(input0Shape[1] != input1Shape[1])
        {
            return;
        }
        if(input0Shape[2] != input1Shape[2])
        {
            return;
        }

        vector<int> outputShape;
        outputShape.push_back(input0Shape[0]);
        outputShape.push_back(input0Shape[1]);
        result->set_shape(outputShape);
        vector<float>* outputData = result->get_data();

        for(int i=0; i<input0Shape[0]; i++)
        {
            for(int j=0; j<input0Shape[1]; j++)
            {
                float sum = 0.0f;
                for(int k=0; k<input0Shape[2]; k++)
                {
                    sum += input0Data->data()[(i*input0Shape[1]+j)*input0Shape[2]+k] * input1Data->data()[j*input1Shape[2]+k];
                }
                outputData->data()[i*input0Shape[1]+j] = sum;
            }
        }
    }

    output[0] = result;
}

int Matmul::CreateInstance(Layer* &layer)
{
    layer = new Matmul();
    return 0;
}

LayerRegistererWrapper matmulCreateInstance("Matmul_t", Matmul::CreateInstance);