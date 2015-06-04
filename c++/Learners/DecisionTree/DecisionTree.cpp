/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#include "../../DataProviders/TreeExamples.h"
#include "DecisionTree.h"

namespace MLx {
    DecisionTreePredictor::DecisionTreePredictor(int dimension, IntVec &ltchild, IntVec &gtChild, IntVec &splitFeatureId, FloatVec &splitThreshold, FloatVec &leafValues)
            : PredictorBase(dimension),
              ltChild_(move(ltchild)), gtChild_(move(gtChild)), splitFeatureId_(move(splitFeatureId)), splitThreshold_(move(splitThreshold)), leafValues_(move(leafValues))
    {}

    float DecisionTreePredictor::Predict(const Vector &features) const {
        return Predict(dynamic_cast<const DenseVector&>(features).Values());
    }

    float DecisionTreePredictor::Predict(const FloatVec &features) const {
        int cur;
        while (cur >= 0)
            cur = features[splitFeatureId_[cur]] < splitThreshold_[cur] ? ltChild_[cur] : gtChild_[cur];
        int leaf = ~cur; // leaf index equals the complement of cur
        assert(0 <= leaf && leaf < leafValues_.size());
        return leafValues_[leaf];
    }

    DecisionTreePredictor DecisionTreeTrainer::TrainCore(const Examples &dataset) {
        auto treeExamples = dynamic_cast<TreeExamples*>(&dataset);
        if (treeExamples == nullptr)
            treeExamples = new TreeExamples(dataset);

        size_t dimension = treeExamples->Schema()->Dimension();
    }
}