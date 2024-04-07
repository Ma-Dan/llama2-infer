#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

using namespace std;

class Tensor
{
    public:
        Tensor();
        ~Tensor();

        int set_shape(const vector<int> shape);
        int load_data(FILE *fp, long offset);
        void set_data(const vector<float> data);
        vector<float>* get_data();
        vector<int> get_shape();

    private:
        vector<int> _shape;
        vector<float> _data;

        void clear();
};

#endif