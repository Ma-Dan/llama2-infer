#include "layer_register.h"

LayerRegister::CreatorRegistry* LayerRegister::_registry = nullptr;

LayerRegister::CreatorRegistry* LayerRegister::Registry()
{
    if(nullptr == _registry)
    {
        _registry = new CreatorRegistry();
    }

    return _registry;
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