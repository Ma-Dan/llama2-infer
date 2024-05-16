#include "unaryop.h"
#include "layer_register.h"
#include "utils.h"
#include "math.h"

UnaryOp::UnaryOp()
{
    _op_type = -1;
}

UnaryOp::~UnaryOp()
{
}

int UnaryOp::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> op_type_param = split(params[6], "=");

    if(op_type_param[1] == "Square")
    {
        _op_type = 0;
    }

    if(op_type_param[1] == "Rsq")
    {
        _op_type = 1;
    }

    return 0;
}

void UnaryOp::forward(vector<Tensor*> &input, vector<Tensor*> &output)
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

    result->set_shape(input[0]->get_shape());
    vector<float>* data = result->get_data();

    switch(_op_type)
    {
        case 0:
            {
                #pragma omp parallel for
                for(int i=0; i<input[0]->get_data()->size(); i++)
                {
                    data->data()[i] = input[0]->get_data()->data()[i] * input[0]->get_data()->data()[i];
                }
            }
            break;
        case 1:
            {
                #pragma omp parallel for
                for(int i=0; i<input[0]->get_data()->size(); i++)
                {
                    data->data()[i] = 1.0f / powf(input[0]->get_data()->data()[i], 0.5f);
                }
            }
            break;
        default:
            break;
    }

    output[0] = result;
}

int UnaryOp::CreateInstance(Layer* &layer)
{
    layer = new UnaryOp();
    return 0;
}

LayerRegistererWrapper unaryOpCreateInstance("UnaryOp_t", UnaryOp::CreateInstance);