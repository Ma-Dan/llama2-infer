#include "node.h"
#include "layer_register.h"

Node::Node(const string &layer_type, const string &node_name)
{
    CreateLayerWrapper create_layer(layer_type, layer);
    this->name = node_name;
}

void Node::forward()
{

}