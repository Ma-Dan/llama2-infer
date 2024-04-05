#ifndef EMBED_H
#define EMBED_H

#include <vector>

#include "layer.h"

class Embed: public Layer
{
    public:
        Embed();
        void forward();

        static int CreateInstance(Layer& layer);
};

#endif