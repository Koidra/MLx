#pragma once

#include <assert.h>
#include <memory>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <smmintrin.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#define UREF std::unique_ptr
#define REF std::shared_ptr

//This logs all exceptions explicitly thrown by the methods. These are considered usage exceptions.
//The variable is defined in ML.cpp
extern std::unordered_set<std::exception*> IsErrorUsage;

namespace MLx {
    typedef std::string string;

    typedef std::vector<float> Vec;
    typedef std::vector<int> IntVec;
    typedef std::vector<UREF<std::string>> StrVec;
    typedef std::vector<bool> BoolVec;

    class FormatException : public std::domain_error {
    public:
        FormatException(const std::string& message);
    };

    namespace Contracts {
        //ToDo: use variadic arguments to pass in other variables for richer exception messages
        template <typename TException>
        void Fail(const std::string& message) {
            TException exception(message);
            IsErrorUsage.insert(&exception);
            throw exception;
        }

        template<typename TException>
        void Check(bool condition, const std::string &message) {
            if (!condition)
                Fail<TException>(message);
        };

        void CheckArg(bool condition, const std::string &message);
    }

    namespace Utils {
        bool IsEmptyOrWhiteSpace(const std::string& s);
        void ToLower(std::string& s);
        void Trim(std::string& s);
        StrVec Split(const std::string& s, char delim);
        StrVec ReadAllLines(const std::string& filename);
    }
}