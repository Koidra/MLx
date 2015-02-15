/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once
#include "Examples.h"

namespace MLx {
    //This class represents examples that are streamed from a text file
    class TextLoader final : public StreamingExamples {
    public:
        TextLoader(const std::string& filename, const std::string& settings);

    protected:
        class TextLoaderState;
    };
}