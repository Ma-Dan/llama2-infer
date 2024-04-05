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
        ~Node();

        int load_model(const vector<string> params, FILE *fp);

        void forward();

    private:
        string name;
        vector<Tensor*> input;
        Tensor* output;
        Layer* layer;
};

#endif