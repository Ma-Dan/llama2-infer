#include "matmul.h"
#include "layer_register.h"
#include "utils.h"
#include "cuda_function.h"
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
    if(_matmul_type == Matmul_GPU)
    {
        freeGPUData(_input_data);
        freeGPUData(_output_data);

        freeGPUData(_input_softmax);
        freeGPUData(_input_v);
        freeGPUData(_output_attention);
    }
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

    if(input[1]->get_device_type() == Tensor_GPU)
    {
        _matmul_type = Matmul_GPU;
        if(_input_data == NULL)
        {
            mallocGPUData(&_input_data, input[0]->get_size()*sizeof(float));
        }

        if(_output_data == NULL)
        {
            mallocGPUData(&_output_data, input0Shape[0]*input1Shape[0]*sizeof(float));
        }
    }

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
                    if(input[1]->has_bias())
                    {
                        sum = input[1]->get_bias()->data()[j];
                    }
                    for(int k=0; k<input0Shape[1]; k++)
                    {
                        sum += input0Data->data()[i*input0Shape[1]+k] * input1Data->data()[j*input1Shape[1]+k];
                    }
                    outputData->data()[i*input1Shape[0]+j] = sum;
                }
            }
        }
        else
        {
            if(input[1]->has_bias())
            {
                matmul_cublas(outputData->data(), input0Data->data(), input[1]->get_device_data(), input[1]->get_bias()->data(), _input_data, _output_data, input0Shape[1], input1Shape[0]);
            }
            else
            {
                matmul_cublas(outputData->data(), input0Data->data(), input[1]->get_device_data(), NULL, _input_data, _output_data, input0Shape[1], input1Shape[0]);
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
        for(i=0; i<input0Shape[0]; i++)
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

        if(0)//_matmul_type == Matmul_GPU)
        {
            if(NULL == _input_softmax)
            {
                mallocGPUData(&_input_softmax, 4096*input0Shape[1]*sizeof(float));
                mallocGPUData(&_input_v, input0Shape[1]*input0Shape[2]*4096*sizeof(float));
                mallocGPUData(&_output_attention, input0Shape[1]*input0Shape[2]*sizeof(float));
            }

            omp_set_max_active_levels(3);
            //#pragma omp parallel for
            for(int i=0; i<input0Shape[1]; i++)
            {
                vector<float> inputSoftmax(input0Shape[0], 0);
                vector<float> inputV(input0Shape[2]*input0Shape[0], 0);
                vector<float> outputAttention(input0Shape[2], 0);
                //head循环
                for(int j=0; j<input0Shape[2]; j++)
                {
                    //head_dim循环
                    for(int k=0; k<input0Shape[0]; k++)
                    {
                        //pos循环
                        inputV[j*input0Shape[0]+k] = input0Data->data()[k*input0Shape[1]*input0Shape[2]+i*input0Shape[2]+j];
                        inputSoftmax[k] = input1Data->data()[k*input0Shape[1]+i];
                    }
                }
                matmul_cublas_qkv(outputAttention.data(), inputV.data(), inputSoftmax.data(), &_input_v[i*input0Shape[2]*4096], &_input_softmax[i*4096], &_output_attention[i*input0Shape[2]], input0Shape[0], input0Shape[2]);
                memcpy(&outputData->data()[i*input0Shape[2]], outputAttention.data(), input0Shape[2]*sizeof(float));
            }
        }
        else
        {
            int i;
            #pragma omp parallel for private(i)
            for(int i=0; i<input0Shape[1]; i++)
            {
                //head循环
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