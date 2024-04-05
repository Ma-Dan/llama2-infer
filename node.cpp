#include "node.h"
#include "layer_register.h"

Node::Node(const string &layer_type, const string &node_name)
{
    CreateLayerWrapper create_layer(layer_type, layer);
    this->name = node_name;
}

Node::~Node()
{
    delete layer;
}

int Node::load_model(const vector<string> params, FILE *fp)
{
    layer->load_model(params, fp);
    return 0;
}

void Node::forward()
{

}