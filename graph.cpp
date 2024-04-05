#include "graph.h"

#include "utils.h"

#include <iostream>
#include <fstream>
#include <string>

Graph::Graph()
{

}

int Graph::load_param(string file_name)
{
    ifstream fin(file_name.c_str());
    int index = 0;
    string strline;
    while (getline(fin, strline) && index < 20)
    {
        if(index > 1)
        {
            vector<string> params = split(strline, " ");
            Node* node = new Node(params[0], params[1]);
            nodes.push_back(node);
            cout << params[0] << endl;
        }
        index ++;
    }
    fin.close();

    return 0;
}