#ifndef GRAPH_H
#define GRAPH_H

#include <vector>

#include "node.h"

class Graph
{
    public:
        Graph();

    private:
        std::vector<Node> nodes;
};

#endif