#ifndef GRAPH_H
#define GRAPH_H

#include <vector>

using namespace std;

#include "node.h"
#include "layer_register.h"

class Graph
{
    public:
        Graph();
        ~Graph();

        int load_param(string file_name);

    private:
        vector<Node*> nodes;
};

#endif