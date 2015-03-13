#pragma once
#include "TextLoader.h"

namespace MLx {
    class DataProviders {
    public:
        static Examples* Load(std::string& fileName, std::string& settings);
    };
}