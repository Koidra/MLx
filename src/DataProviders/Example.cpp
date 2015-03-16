#include "Example.h"

namespace MLx {
    using namespace std;

    Example::Example(Vector* features, float label, float weight, UREF<string> name)
            : Features(UREF<Vector>(features)), Label(label), Weight(weight), Name(move(name)) {}
}