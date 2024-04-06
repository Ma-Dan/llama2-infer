#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>

using namespace std;

#include "tensor.h"
#include "layer.h"

class Graph;

class Node
{
    public:
        Node(Graph* graph, const vector<string> params);
        ~Node();

        int load_model(const vector<string> params, FILE *fp);
        void forward();
        vector<string> get_input_names();

    private:
        Graph* _graph;
        string _name;
        vector<string> _input_names;
        vector<string> _output_names;
        vector<Tensor*> _inputs;
        vector<Tensor*> _outputs;
        Layer* _layer;
};

#endif