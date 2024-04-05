#include "graph.h"

#include "utils.h"

#include <iostream>
#include <fstream>
#include <string>

Graph::Graph()
{

}

Graph::~Graph()
{
    for(int i=0; i<nodes.size(); i++)
    {
        delete nodes[i];
    }

    nodes.clear();
}

int Graph::load_param(string file_name)
{
    string param_file_name = file_name + ".param";
    string bin_file_name = file_name + ".bin";

    ifstream fin(param_file_name.c_str());

    FILE *fp = fopen(bin_file_name.c_str(), "rb");

    int index = 0;
    string strline;
    while (getline(fin, strline) && index < 20)
    {
        if(index > 1)
        {
            vector<string> params = split(strline, " ");
            Node* node = new Node(params[0], params[1]);
            node->load_model(params, fp);
            nodes.push_back(node);
            cout << params[0] << endl;
        }
        index ++;
    }
    fin.close();

    fclose(fp);

    return 0;
}