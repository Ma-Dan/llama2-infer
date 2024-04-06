#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>

using namespace std;

#include "layer_register.h"

class Node;

class Graph
{
    public:
        Graph();
        ~Graph();

        int load_model(string file_name);
        void register_operand(string operand_name);
        void register_operand(string operand_name, Tensor* operand_tensor);
        void register_output(string operand_name, Node* node);

        void input(string operand_name, Tensor* input_tensor);
        int extract(string operand_name, Tensor* &output_tensor);

        Tensor* find_operand(string operand_name);

    private:
        vector<Node*> _nodes;
        map<string, Tensor*> _operand_map;
        map<string, vector<Node*>> _exec_order_map;
        map<string, Node*> _output_owner_map;

        int topo_sort(string operand_name, vector<Node*> &exec_list);
};

#endif