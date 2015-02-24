/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once

#include "../Trainer.h"

namespace MLx {
    class DecisionTreePredictor : public PredictorBase<float> {
    public:
        virtual float Predict(const Vector &features) const final;
        virtual float Predict(const FloatVec &features) const final;
    protected:
        DecisionTreePredictor(int dimension, IntVec &ltchild, IntVec &gtChild, IntVec &splitFeatureId, FloatVec &splitThreshold, FloatVec &leafValues);
    private:
        /* Tree structure
         *  - Every node has 2 children, which are either other nodes or leaves
         *  - - Note: node means a non-leaf and a leaf in this context is not a node
         *  - - - Hence: number of nodes equals number of leaves - 1
         *  - Every node has corresponding feature index and split threshold
         */

        //Defined for non-leaf nodes
        IntVec ltChild_, gtChild_;
        IntVec splitFeatureId_;
        FloatVec splitThreshold_;

        //Defined for leaves
        FloatVec leafValues_; //REVIEW: should be distribution
    };

    class DecisionTreeTrainer : public TrainerBase<DecisionTreePredictor> {
    public:
        DecisionTreeTrainer(const string& settings);
    protected:
        DecisionTreePredictor TrainCore(const Examples& dataset) override;
    };
}