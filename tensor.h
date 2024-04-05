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
        int load_data(FILE *fp);

    private:
        std::vector<int> shape_;
        std::vector<float> data_;

        void clear();
};

#endif