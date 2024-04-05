#ifndef INPUT_H
#define INPUT_H

#include <vector>

#include "layer.h"

class Input: public Layer
{
    public:
        Input();
        void forward();

        static int CreateInstance(Layer& layer);
};

#endif