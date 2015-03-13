#pragma once
#include "Examples.h"

namespace MLx {
    //This class represents examples that are streamed from a text file
    class TextLoader final : public Examples {
    public:
        TextLoader(const std::string& filename, const std::string& settings);
        ExamplesIterator* begin() override;
        ExamplesIterator* end() override;

    private:
        class Impl;
        UREF<Impl> impl_;
    };
}