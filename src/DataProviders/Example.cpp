#include "Example.h"

namespace MLx {
    using namespace std;

    Example::Example(Vector* features, float label, float weight, UREF<string> name)
            : features_(UREF<Vector>(features)), Label(label), Weight(weight), name_(move(name)) {}

    const Vector& Example::Features() const {
        return *features_;
    }

    const string& Example::Name() const {
        return *name_;
    }
}