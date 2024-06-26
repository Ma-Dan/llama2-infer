#include "matmul.h"
#include "layer_register.h"
#include "utils.h"
#include "omp.h"

Matmul::Matmul()
{
    _matmul_type = Matmul_CPU;
    _input_data = NULL;
    _output_data = NULL;

    _input_softmax = NULL;
    _input_v = NULL;
    _output_attention = NULL;
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

    if(input0Shape.size() == 2 && input1Shape.size() == 2)
    {
        //Linear
        if(input0Shape[1] != input1Shape[1])
        {
            return;
        }

        vector<int> outputShape;
        outputShape.push_back(input0Shape[0]);
        outputShape.push_back(input1Shape[0]);
        result->set_shape(outputShape);
        vector<float>* outputData = result->get_data();

        if(_matmul_type == Matmul_CPU)
        {
            int i;
            #pragma omp parallel for private(i)
            for(i=0; i<input0Shape[0]; i++)
            {
                int j;
                #pragma omp parallel for private(j)
                for(j=0; j<input1Shape[0]; j++)
                {
                    float sum = 0.0f;
                    //#pragma omp parallel for reduction( +:sum)
                    for(int k=0; k<input0Shape[1]; k++)
                    {
                        sum += input0Data->data()[i*input0Shape[1]+k] * input1Data->data()[j*input1Shape[1]+k];
                    }
                    outputData->data()[i*input1Shape[0]+j] = sum;
                }
            }
        }
    }
    else if(input0Shape.size() == 3 && input1Shape.size() == 3)
    {
        //QK
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

        int i;
        #pragma omp parallel for private(i)
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
    else if(input0Shape.size() == 3 && input1Shape.size() == 2)
    {
        //QKV
        //TODO:引入transpose不特殊处理
        vector<int> outputShape;
        outputShape.push_back(1);
        outputShape.push_back(input0Shape[1]);
        outputShape.push_back(input0Shape[2]);
        result->set_shape(outputShape);
        vector<float>* outputData = result->get_data();

        if(1)//_matmul_type == Matmul_GPU)
        {
            int i;
            #pragma omp parallel for private(i)
            for(i=0; i<input0Shape[1]; i++)
            {
                //head循环
                int j;
                for(int j=0; j<input0Shape[2]; j++)
                {
                    //head_dim循环
                    float sum = 0.0f;
                    for(int k=0; k<input0Shape[0]; k++)
                    {
                        //pos循环
                        sum += input0Data->data()[k*input0Shape[1]*input0Shape[2]+i*input0Shape[2]+j]*input1Data->data()[k*input0Shape[1]+i];
                    }
                    outputData->data()[i*input0Shape[2]+j] = sum;
                }
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