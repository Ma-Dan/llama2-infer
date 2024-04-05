#ifndef LAYER_REGISTER_H
#define LAYER_REGISTER_H

#include <vector>
#include <map>
#include <string>

using namespace std;

#include "layer.h"

class LayerRegister
{
    public:
        typedef int (*Creator)(Layer& layer);
        typedef map<string, Creator> CreatorRegistry;
        static CreatorRegistry *registry_;

        static CreatorRegistry* Registry();
        static void RegisterCreator(const string& layer_type, const Creator &creator);
        static void CreateLayer(const string& layer_type, Layer& layer);
};

class LayerRegistererWrapper
{
    public:
        explicit LayerRegistererWrapper(const string& layer_type, const LayerRegister::Creator& creator)
        {
            LayerRegister::RegisterCreator(layer_type, creator);
        }
};

class CreateLayerWrapper
{
    public:
        explicit CreateLayerWrapper(const string& layer_type, Layer& layer)
        {
            LayerRegister::CreateLayer(layer_type, layer);
        }
};

#endif