/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once
#include "TextLoader.h"

namespace MLx {
    namespace DataProviders {
        Examples* Load(std::string& fileName, std::string& settings);
    };
}