#include "Core/Commons.h"
#include "DataProviders/DataProviders.h"

using namespace std;
using namespace MLx;

std::unordered_set<std::exception*> IsErrorUsage;

int main(int argc, char* argv[]) {
    string trainFile(argv[1]);
    string testFile(argv[2]);
    string settings("");

    auto trainExamples = DataProviders::Load(trainFile, settings);
    return 0;
}