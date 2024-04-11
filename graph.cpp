#include "node.h"
#include "graph.h"

#include "utils.h"

#include <iostream>
#include <fstream>
#include <string>
#include <set>

Graph::Graph()
{

}

Graph::~Graph()
{
    for(int i=0; i<_nodes.size(); i++)
    {
        SAFE_DELETE(_nodes[i]);
    }

    _nodes.clear();
}

int Graph::load_model(string file_name)
{
    string param_file_name = file_name + ".param";
    string bin_file_name = file_name + ".bin";

    ifstream fin(param_file_name.c_str());
    FILE *fp = fopen(bin_file_name.c_str(), "rb");

    int index = 0;
    string strline;
    while(getline(fin, strline))
    {
        if(index > 1)
        {
            vector<string> params = split(strline, " ");
            Node* node = new Node(this, params);
            node->load_model(params, fp);
            _nodes.push_back(node);
        }
        index++;
    }

    fin.close();
    fclose(fp);

    return 0;
}

void Graph::register_operand(string operand_name)
{
    if(_operand_map.find(operand_name) == _operand_map.end())
    {
        _operand_map[operand_name] = nullptr;
    }
}

void Graph::register_operand(string operand_name, Tensor* operand_tensor)
{
    if(_operand_map.find(operand_name) == _operand_map.end())
    {
        _operand_map[operand_name] = operand_tensor;
        return;
    }

    if(_operand_map[operand_name] == nullptr)
    {
        _operand_map[operand_name] = operand_tensor;
        return;
    }

    if(_operand_map[operand_name] == operand_tensor)
    {
        return;
    }

    SAFE_DELETE(_operand_map[operand_name]);
    _operand_map[operand_name] = operand_tensor;
}

void Graph::register_output(string operand_name, Node* node)
{
    if(_output_owner_map.find(operand_name) == _output_owner_map.end())
    {
        _output_owner_map[operand_name] = node;
    }
}

void Graph::input(string operand_name, Tensor* input_tensor)
{
    if(_operand_map.find(operand_name) != _operand_map.end())
    {
        if(_operand_map[operand_name] != nullptr)
        {
            //SAFE_DELETE(_operand_map[operand_name]);
        }
    }

    _operand_map[operand_name] = input_tensor;
}

int Graph::extract(string operand_name, Tensor* &output_tensor)
{
    if(_operand_map.find(operand_name) == _operand_map.end())
    {
        return -1;
    }

    //检查有无该输出的执行列表
    if(_exec_order_map.find(operand_name) == _exec_order_map.end())
    {
        //没有则执行拓扑排序得到节点列表
        vector<Node*> execOrder;
        if(0 != topo_sort(operand_name, execOrder))
        {
            return -1;
        }
        _exec_order_map[operand_name] = execOrder;
    }

    //执行列表
    vector<Node*> execList = _exec_order_map[operand_name];
    for(int i=0; i<execList.size(); i++)
    {
        execList[i]->forward();
    }

    output_tensor = _operand_map[operand_name];

    return 0;
}

int Graph::get_result(string operand_name, Tensor* &output_tensor)
{
    if(_operand_map.find(operand_name) == _operand_map.end())
    {
        return -1;
    }

    output_tensor = _operand_map[operand_name];

    return 0;
}

Tensor* Graph::find_operand(string operand_name)
{
    return _operand_map[operand_name];
}

int Graph::topo_sort(string operand_name, vector<Node*> &exec_list)
{
    Node* outputNode = _output_owner_map[operand_name];

    map<Node*, set<Node*>> nodeIndegrees;

    for(int i=0; i<_nodes.size(); i++)
    {
        set<Node*> indegree;
        vector<string> inputNames = _nodes[i]->get_input_names();
        for(int j=0; j<inputNames.size(); j++)
        {
            Node* inputNode = _output_owner_map[inputNames[j]];
            indegree.insert(inputNode);
        }

        nodeIndegrees[_nodes[i]] = indegree;
    }

    while(1)
    {
        set<Node*> nodesToRemove;
        map<Node*, set<Node*>>::iterator iterCheck;
        for(iterCheck=nodeIndegrees.begin(); iterCheck != nodeIndegrees.end(); iterCheck++)
        {
            if(iterCheck->second.size() == 0)
            {
                Node* nodeRemove = iterCheck->first;
                exec_list.push_back(nodeRemove);
                nodesToRemove.insert(nodeRemove);
            }
        }

        if(nodesToRemove.size()==0 && nodeIndegrees.find(outputNode)!=nodeIndegrees.end())
        {
            //没有找到执行到该节点的路径
            return -1;
        }

        set<Node*>::iterator iterRemove;
        for(iterRemove=nodesToRemove.begin(); iterRemove != nodesToRemove.end(); iterRemove++)
        {
            map<Node*, set<Node*>>::iterator iterNode;
            for(iterNode=nodeIndegrees.begin(); iterNode != nodeIndegrees.end(); iterNode++)
            {
                iterNode->second.erase(*iterRemove);
            }

            nodeIndegrees.erase(*iterRemove);
        }

        if(nodeIndegrees.find(outputNode) == nodeIndegrees.end())
        {
            return 0;
        }
    }

    return -1;
}