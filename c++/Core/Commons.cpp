/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#include "Commons.h"

namespace MLx {
    using namespace std;

    std::mt19937 Rand(1);

    FormatException::FormatException(const string& message) : domain_error(message) {};DfDiffpFile

    namespace Utils {
        using namespace Contracts;

        bool IsEmptyOrWhiteSpace(const string& s) {
            return s.find_first_not_of(" \t") == s.npos;
        }

        void ToLower(string &s) {
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        }

        void Trim(string &s) {
            boost::algorithm::trim(s);
        }

        StrVec Split(const string &s, char delim) {
            stringstream ss(s);
            StrVec tokens;
            auto item = UREF<string>(new string());
            while (getline(ss, *item, delim)) {
                tokens.push_back(move(item));
                item = UREF<string>(new string());
            }
            return tokens;
        }

        StrVec ReadAllLines(const string& filename) {
            ifstream stream(filename);
            CheckArg(stream.good(), "File not found");
            StrVec lines;
            auto line = UREF<string>(new string());
            while (getline(stream, *line)) {
                lines.push_back(move(line));
                line = UREF<string>(new string());
            }
            return lines;
        }
    }
}