#include "layer_register.h"

LayerRegister::CreatorRegistry* LayerRegister::registry_ = nullptr;

LayerRegister::CreatorRegistry* LayerRegister::Registry()
{
    if(nullptr == registry_)
    {
        registry_ = new CreatorRegistry();
    }

    return registry_;
}

void LayerRegister::RegisterCreator(const string& layer_type, const Creator &creator)
{
    CreatorRegistry* registry = Registry();
    registry->insert({layer_type, creator});
}

void LayerRegister::CreateLayer(const string& layer_type, Layer* &layer)
{
    CreatorRegistry* registry = Registry();
    (*registry)[layer_type](layer);
}