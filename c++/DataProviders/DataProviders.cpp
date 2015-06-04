/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#include "DataProviders.h"

namespace MLx {
    using namespace std;
    using namespace Utils;
    using namespace Contracts;

    namespace DataProviders {
        Examples* DataProviders::Load(string& fileName, string& settings) {
            string fileExtension  = boost::filesystem::extension(fileName);
            ToLower(fileExtension);

            if (fileExtension == ".txt" || fileExtension == ".tsv")
                return new TextLoader(fileName, settings);
            if (fileExtension == ".csv")
                return new TextLoader(fileName, "sep:," + settings);

            CheckArg(false, "Unknown file format: " + fileExtension);
            return nullptr;
        }
    }
}