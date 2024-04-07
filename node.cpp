#include "node.h"
#include "layer_register.h"
#include "graph.h"
#include "utils.h"

Node::Node(Graph* graph, const vector<string> params)
{
    _graph = graph;

    _name = params[1];
    CreateLayerWrapper create_layer(params[0], _layer);

    int input_count = atoi(params[2].c_str());
    int output_count = atoi(params[3].c_str());

    for(int i=0; i<input_count; i++)
    {
        string inputName = params[4+i];
        _input_names.push_back(inputName);
        _graph->register_operand(inputName);
    }

    for(int i=0; i<output_count; i++)
    {
        string outputName = params[4+input_count+i];
        _output_names.push_back(outputName);
        _graph->register_operand(outputName);
        _graph->register_output(outputName, this);
    }
}

Node::~Node()
{
    SAFE_DELETE(_layer);
}

int Node::load_model(const vector<string> params, FILE *fp)
{
    _layer->load_model(params, fp);
    return 0;
}

void Node::forward()
{
    vector<Tensor*> inputTensors;
    vector<Tensor*> outputTensors;

    for(int i=0; i<_input_names.size(); i++)
    {
        inputTensors.push_back(_graph->find_operand(_input_names[i]));
    }

    for(int i=0; i<_output_names.size(); i++)
    {
        outputTensors.push_back(_graph->find_operand(_output_names[i]));
    }

    _layer->forward(inputTensors, outputTensors);

    for(int i=0; i<_output_names.size(); i++)
    {
        _graph->register_operand(_output_names[i], outputTensors[i]);
    }
}

vector<string> Node::get_input_names()
{
    return _input_names;
}