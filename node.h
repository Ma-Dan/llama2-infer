#ifndef NODE_H
#define NODE_H

#include <string>
#include <vector>

using namespace std;

#include "tensor.h"
#include "layer.h"

class Node
{
    public:
        Node(const string &layer_type, const string &node_name);

        void forward();

    private:
        string name;
        vector<Tensor> input;
        Tensor output;
        Layer layer;
};

#endif