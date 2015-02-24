#include "Examples.h"

namespace MLx {
    using namespace std;
    using namespace Contracts;
    using namespace Utils;

    DataSchema::DataSchema(StrVec& featureNames) : FeatureNames(move(featureNames)) {
    }

    size_t DataSchema::Dimension() const {
        return FeatureNames.size();
    }

    bool Examples::IsSparse() {
        return isSparse_;
    }

    const DataSchema* Examples::Schema() const {
        return &*schema_;
    }

    Examples::Iterator Examples::begin() const {
        state_->Reset();
        return Iterator(&*state_);
    }

    auto NullIter = Examples::Iterator(nullptr);
    Examples::Iterator Examples::end() const {
        return NullIter;
    }

    Examples::Iterator::Iterator(const State *state) : state_(state) {}

    const Example& Examples::Iterator::operator*() const {
        return *(state_->Current());
    }

    const Example* Examples::Iterator::operator->() const {
        return state_->Current();
    }

    Examples::Iterator Examples::Iterator::operator++() {
        return  state_->MoveNext() ? Iterator(state_) : NullIter;
    }

    bool Examples::Iterator::operator!=(const Iterator &other) {
        return other.state_ != state_;
    }

    class InMemoryExamples::State final : public Examples::State {
        const std::vector<Example>* data_;
        std::vector<Example>::const_iterator iter_;
    public:
        State(std::vector<Example>* data) : data_(data), iter_(data->begin()) { }

        void Reset() override {
            iter_ = data_->begin();
        }

        bool MoveNext() override {
            return ++iter_ != data_->end();
        }

        const Example* Current() const override {
            return &*iter_;
        }
    };

    InMemoryExamples::InMemoryExamples(REF<DataSchema> schema, bool isSparse, std::vector<Example> &data)
            : data_(move(data))
    {
        schema_ = schema;
        isSparse_ = isSparse;
        state_ = UREF<State>(new InMemoryExamples::State(&data_));
    }

    size_t InMemoryExamples::Size() {
        return data_.size();
    }

    void InMemoryExamples::Add(UREF<Example> example) {
        data_.push_back(move(*example));
    }

    StreamingExamples::operator InMemoryExamples() {
        return  InMemoryExamples(schema_, isSparse_, ((StreamingExamples::State &)(*state_)).Cache());
    }

    StreamingExamples::State::~State() {
        if (current_ != nullptr && cache_.empty())
            delete current_;
    }

    const Example* StreamingExamples::State::Current() const {
        return current_;
    }

    vector<Example>& StreamingExamples::State::Cache() {
        if (cache_.empty())
        {
            cache_.clear();
            Reset();
            do{
                cache_.push_back(move(*current_));
            } while (MoveNext());
        }
        return cache_;
    }
}