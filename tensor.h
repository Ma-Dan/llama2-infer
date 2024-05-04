#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory.h>
#include <stdio.h>

using namespace std;

enum
{
    Tensor_CPU = 0,
    Tensor_GPU,
};

class Tensor
{
    public:
        Tensor();
        ~Tensor();

        int set_shape(const vector<int> shape);
        int set_bias(int bias);
        int load_data(FILE *fp, long offset);
        void set_data(const vector<float> data);
        vector<float>* get_data();
        vector<float>* get_bias();
        vector<int> get_shape();
        int has_bias();
        int get_size();

        int get_device_type();
        void set_device_type(int device_type);
        float* get_device_data();
        float* get_device_bias();

    private:
        vector<int> _shape;
        vector<float> _data;
        vector<float> _bias;

        int _has_bias;

        void clear();

        int _device_type;
        float* _device_data;
        float* _device_bias;
};

#endif