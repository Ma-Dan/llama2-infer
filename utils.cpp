#include <vector>
#include <string>

using namespace std;

vector<string> split(const string& str, const string& delim)
{
    vector<string> result;

    if("" == str)
    {
        return result;
    }

    string strs = str + delim;
    size_t pos;
    size_t size = strs.size();

    for (int i = 0; i < size; ++i)
    {
        pos = strs.find(delim, i);
        if(pos < size)
        {
            string s = strs.substr(i, pos - i);
            result.push_back(s);
            i = pos + delim.size() - 1;
        }
    }

    return result;
}

bool is_same_shape(const vector<int> shape1, const vector<int> shape2)
{
    if(shape1.size() != shape2.size())
    {
        return false;
    }

    for(int i=0; i<shape1.size(); i++)
    {
        if(shape1[i] != shape2[i])
        {
            return false;
        }
    }

    return true;
}