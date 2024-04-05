#include "embed.h"
#include "layer_register.h"

Embed::Embed()
{

}

void Embed::forward()
{

}

int Embed::CreateInstance(Layer& layer)
{
    return 0;
}

LayerRegistererWrapper embedCreateInstance("Embed_t", Embed::CreateInstance);