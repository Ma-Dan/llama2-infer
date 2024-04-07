#include "binaryop.h"
#include "layer_register.h"
#include "utils.h"

BinaryOp::BinaryOp()
{
}

BinaryOp::~BinaryOp()
{
}

int BinaryOp::load_model(const vector<string> &params, FILE* fp)
{
    int inputCount = atoi(params[2].c_str());
    int outputCount = atoi(params[3].c_str());

    vector<string> op_type_param = split(params[4+inputCount+outputCount], "=");
    if(op_type_param[1] == "Add")
    {
        _op_type = 0;
    }
    else if (op_type_param[1] == "Sub")
    {
        _op_type = 1;
    }
    else if (op_type_param[1] == "Mul")
    {
        _op_type = 2;
    }
    else if (op_type_param[1] == "Div")
    {
        _op_type = 3;
    }

    vector<string> param_type_param = split(params[5+inputCount+outputCount], "=");
    if(param_type_param[1] == "Param")
    {
        vector<string> param_value_param = split(params[6+inputCount+outputCount], "=");
        _param = atof(param_value_param[1].c_str());

        _param_type = 0;
    }
    else if(param_type_param[1] == "Tensor")
    {
        _param_type = 1;
    }

    return 0;
}

void BinaryOp::forward(vector<Tensor*> &input, vector<Tensor*> &output)
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

    if(_param_type == 0)
    {
        result->set_shape(input[0]->get_shape());
        vector<float>* inputData = input[0]->get_data();
        vector<float>* outputData = result->get_data();

        switch(_op_type)
        {
            case 0:
                {
                    for(int i=0; i<outputData->size(); i++)
                    {
                        outputData->data()[i] = inputData->data()[i] + _param;
                    }
                }
                break;
            case 1:
                {
                    for(int i=0; i<outputData->size(); i++)
                    {
                        outputData->data()[i] = inputData->data()[i] - _param;
                    }
                }
                break;
            case 2:
                {
                    for(int i=0; i<outputData->size(); i++)
                    {
                        outputData->data()[i] = inputData->data()[i] * _param;
                    }
                }
                break;
            case 3:
                {
                    for(int i=0; i<outputData->size(); i++)
                    {
                        outputData->data()[i] = inputData->data()[i] / _param;
                    }
                }
                break;
            default:
                break;
        }

        output[0] = result;
        return;
    }

    if(_param_type == 1)
    {
        //简单分为两种情况
        //形状不同 input0 乘以 input1的第一个元素
        //形状相同 逐个相乘
        if(!is_same_shape(input[0]->get_shape(), input[1]->get_shape()))
        {
            result->set_shape(input[0]->get_shape());
            vector<float>* input0Data = input[0]->get_data();
            vector<float>* input1Data = input[1]->get_data();
            vector<float>* outputData = result->get_data();

            float param = input1Data->data()[0];

            switch(_op_type)
            {
                case 0:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] + param;
                        }
                    }
                    break;
                case 1:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] - param;
                        }
                    }
                    break;
                case 2:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] * param;
                        }
                    }
                    break;
                case 3:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] / param;
                        }
                    }
                    break;
                default:
                    break;
            }

            output[0] = result;
        }
        else
        {
            result->set_shape(input[0]->get_shape());
            vector<float>* input0Data = input[0]->get_data();
            vector<float>* input1Data = input[1]->get_data();
            vector<float>* outputData = result->get_data();

            switch(_op_type)
            {
                case 0:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] + input1Data->data()[i];
                        }
                    }
                    break;
                case 1:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] - input1Data->data()[i];
                        }
                    }
                    break;
                case 2:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] * input1Data->data()[i];
                        }
                    }
                    break;
                case 3:
                    {
                        for(int i=0; i<outputData->size(); i++)
                        {
                            outputData->data()[i] = input0Data->data()[i] / input1Data->data()[i];
                        }
                    }
                    break;
                default:
                    break;
            }

            output[0] = result;
        }
    }
}

int BinaryOp::CreateInstance(Layer* &layer)
{
    layer = new BinaryOp();
    return 0;
}

LayerRegistererWrapper binaryOpCreateInstance("BinaryOp_t", BinaryOp::CreateInstance);