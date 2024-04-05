#include "layer.h"

Layer::Layer()
{

}

Layer::~Layer()
{

}

int Layer::load_model(const vector<string> &params, FILE* fp)
{
    return 0;
}

void Layer::forward(vector<Tensor*> &input, Tensor* output)
{

}

int Layer::CreateInstance(Layer* &layer)
{
    return 0;
}