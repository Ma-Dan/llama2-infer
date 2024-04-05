#include "input.h"
#include "layer_register.h"

Input::Input()
{

}

void Input::forward()
{

}

int Input::CreateInstance(Layer& layer)
{
    return 0;
}

LayerRegistererWrapper inputCreateInstance("Input_t", Input::CreateInstance);