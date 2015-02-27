/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once

#include "../DataProviders/Examples.h"
#include "Predictor.h"

namespace MLx {
    template<class TPredictor>
    class TrainerBase {
    protected:
        virtual TPredictor TrainCore(const Examples &dataset) = 0;

    public:
        TPredictor Train(const Examples &dataset) {
            //Print settings

            //ToDo: normalization goes here

            time_t time(0);
            struct tm * now = localtime(&time);
            std::cout << (now->tm_year + 1900) << '/' << (now->tm_mon + 1) << '/' <<  now->tm_mday
                    << " **************** Training ****************" << std::endl;

            return Train(dataset);
        }
    };
}
