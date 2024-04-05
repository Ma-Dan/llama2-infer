#include "embed.h"
#include "layer_register.h"
#include "utils.h"

Embed::Embed()
{
    weight = new Tensor();
}

Embed::~Embed()
{
    delete weight;
}

int Embed::load_model(const vector<string> &params, FILE* fp)
{
    vector<int> shape;
    for(int i=0; i<2; i++)
    {
        vector<string> dim_size = split(params[6+i], "=");
        shape.push_back(atoi(dim_size[1].c_str()));
    }

    weight->set_shape(shape);

    weight->load_data(fp);

    return 0;
}

void Embed::forward(vector<Tensor*> &input, Tensor* output)
{

}

int Embed::CreateInstance(Layer* &layer)
{
    layer = new Embed();
    return 0;
}

LayerRegistererWrapper embedCreateInstance("Embed_t", Embed::CreateInstance);