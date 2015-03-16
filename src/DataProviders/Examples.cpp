#include "Examples.h"

namespace MLx {
    using namespace std;
    using namespace Contracts;
    using namespace Utils;

    DataSchema::DataSchema(StrVec& featureNames) : FeatureNames(move(featureNames)) {
    }

    size_t DataSchema::GetDimension() {
        return (FeatureNames).size();
    }

    bool Examples::IsSparse() {
        return isSparse_;
    }

    DataSchema* Examples::GetSchema() {
        return &*schema_;
    }

    ExampleIterator Examples::begin() const {
        state_->Reset();
        return ExampleIterator(&*state_);
    }

    ExampleIterator NullIter = ExampleIterator(nullptr);
    ExampleIterator Examples::end() const {
        return NullIter;
    }

    ExampleIterator::ExampleIterator(ExamplesReadState *state) : state_(state) {}

    const Example& ExampleIterator::operator*() const {
        return *(state_->Current());
    }

    const Example* ExampleIterator::operator->() const {
        return state_->Current();
    }

    ExampleIterator ExampleIterator::operator++() {
        return  state_->MoveNext() ? ExampleIterator(state_) : NullIter;
    }

    bool ExampleIterator::operator!=(const ExampleIterator &other) {
        return other.state_ != state_;
    }

    InMemoryExamples::InMemoryExamples(Examples &src) {
//        schema_ = move(src.schema_);
//        isSparse_ = src.isSparse_;
//        for (ExampleIterator iter = src.begin(); iter != src.end(); ++iter)
//            data_.push_back(move(*iter));
//        state_ = UREF<ExamplesReadState>(new InMemoryExamples::State(data_));
//        src.state_ = nullptr;
    }

    bool InMemoryExamples::IsEmpty() {
        return data_.empty();
    }

    size_t InMemoryExamples::Size() {
        return data_.size();
    }

    InMemoryExamples::State::State(std::vector<Example> &data)
            : data_(&data), iterator_(data.begin()) {}

    void InMemoryExamples::State::Reset() {
        iterator_ = data_->begin();
    }

    bool InMemoryExamples::State::MoveNext() {
        return ++iterator_ != data_->end();
    }

    const Example* InMemoryExamples::State::Current() const {
        return &(*iterator_);
    }

    REF<InMemoryExamples> StreamingExamples::Cache() {
        throw new runtime_error("Initialize cached and Cache would destroy the current streaming dataset.");
//        if (cached_ != nullptr)
//        {
////            for (auto iter = begin(); iter != end(); ++iter)
////                cached_->data_.push_back(move(*iter));
//        }
//        return cached_;
    }
}