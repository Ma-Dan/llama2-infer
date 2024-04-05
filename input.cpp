#include "input.h"
#include "layer_register.h"

Input::Input()
{

}

void Input::forward(vector<Tensor*> &input, Tensor* output)
{

}

int Input::CreateInstance(Layer* &layer)
{
    layer = new Input();
    return 0;
}

LayerRegistererWrapper inputCreateInstance("Input_t", Input::CreateInstance);