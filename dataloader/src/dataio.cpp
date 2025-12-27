#include "dataio.h"
#include <utility>
#include <vector>
#include <filesystem>
#include <format>
#include <utils.h>

bool checkTransformOperatesOnStandardNDShape(const std::vector<size_t> &inputShape,
                                             std::vector<size_t> &outputShape, size_t n) {
    if (inputShape.size() != 3 || outputShape.size() != 3) {
        return false;
    }

    if (inputShape[2] != n) {
        return false;
    }

    if (!std::equal(inputShape.begin(), inputShape.end(), outputShape.begin())) {
        return false;
    }

    return true;
}

Dataset::Dataset(std::shared_ptr<IDataSource> _dataSource,
                 std::vector<IDataAugmentation *> _dataAugmentations,
                 const pybind11::function &createDatasetFunction,
                 const bool isVirtualDataset
) : dataSource(std::move(_dataSource)), dataAugmentations(std::move(_dataAugmentations)) {
    if (dataSource->preInitDataset(existsEnvVar(INVALID_DS_ENV_VAR) && isVirtualDataset)) {
        createDatasetFunction();
    }

    dataSource->initDataset();
}

Dataset::Dataset(std::shared_ptr<IDataSource> _dataSource,
                 std::vector<IDataAugmentation *> _dataAugmentations)
    : dataSource(std::move(_dataSource)), dataAugmentations(std::move(_dataAugmentations)) {
    dataSource->initDataset();

    // Defensive programming.
    const auto &itemKeys = dataSource->getItemKeys();
    bool rasterEncountered = false;
    for (const auto & itemKey : itemKeys) {
        rasterEncountered |= itemKey.type == ItemType::RASTER;
        if (itemKey.type == ItemType::POINTS) {
            if (!rasterEncountered) {
                throw std::runtime_error("ItemKey of type POINTS must be preceeded by type RASTER to latch onto it.");
            }
            break;
        }
    }
}

std::tuple<Dataset, Dataset, Dataset> Dataset::splitTrainValidationTest(
    const float trainPercentage, const float validPercentage) const {
    if (trainPercentage <= 0.0f || validPercentage <= 0.0f) {
        throw std::runtime_error(
            "Train and validation set must contain more than 0% of elements.");
    }

    const size_t numEntries = dataSource->getEntries().size();
    const int numTrain = static_cast<int>(std::round(trainPercentage * static_cast<float>(numEntries)));
    const int numValid = static_cast<int>(std::round(validPercentage * static_cast<float>(numEntries)));

    if (numTrain + numValid > static_cast<int>(numEntries)) {
        throw std::runtime_error(
            "Violated #train examples + #validation examples <= #all examples.");
    }

    std::shared_ptr<IDataSource> trainSource;
    std::shared_ptr<IDataSource> validSource;
    std::shared_ptr<IDataSource> testSource;
    dataSource->splitIntoTwoDataSources(numTrain, trainSource, validSource);
    validSource->splitIntoTwoDataSources(numValid, validSource, testSource);

    return std::make_tuple<>(
        Dataset(trainSource, dataAugmentations),
        Dataset(validSource, dataAugmentations),
        Dataset(testSource, dataAugmentations)
    );
}

IDataSource *Dataset::getDataSource() const {
    return dataSource.get();
}

std::vector<IDataAugmentation *> Dataset::getDataAugmentations() const {
    return dataAugmentations;
}

BatchedDataset::BatchedDataset(const Dataset &dataset, const size_t batchSize) : dataset(dataset),
    batchSize(batchSize), currInFlightBatch(0), lastWaitingBatch(-static_cast<int>(batchSize)) {
}

BatchedDataset::BatchedDataset(const Dataset &&dataset, const size_t batchSize) : dataset(dataset),
    batchSize(batchSize), currInFlightBatch(0), lastWaitingBatch(-static_cast<int>(batchSize)) {
}

DatasetBatch BatchedDataset::getNextInFlightBatch() {
    std::unique_lock lock(mutex);
    IDataSource *source = dataset.getDataSource();
    const auto &entries = source->getEntries();

    std::vector<std::vector<std::string> > batchPaths;

    const int32_t offset = currInFlightBatch.fetch_add(static_cast<int32_t>(batchSize));
    for (size_t i = offset; i < offset + batchSize; i++) {
        std::vector<std::string> entry;

        for (const auto &subPath: entries[i % entries.size()]) {
            entry.push_back(subPath);
        }

        batchPaths.push_back(std::move(entry));
    }

    inFlightBatches.emplace(offset);
    LOG_DEBUG("getNextInFlightBatch: lastWaitingBatch={}, offset={}", lastWaitingBatch.load(), offset);
    return {
        .startingOffset = offset,
        .batchPaths = std::move(batchPaths)
    };
}

std::vector<std::vector<std::string> > BatchedDataset::getNextBatch() {
    return getNextInFlightBatch().batchPaths;
}

void BatchedDataset::markBatchWaiting(const int32_t batch) {
    std::unique_lock lock(mutex);
    LOG_DEBUG("markBatchWaiting: batch={}", batch);
    lastWaitingBatch = batch;
}

void BatchedDataset::popWaitingBatch(const int32_t batch) {
    std::unique_lock lock(mutex);
    LOG_DEBUG("popWaitingBatch: batch={}", batch);
    inFlightBatches.erase(batch);
}

void BatchedDataset::forgetInFlightBatches() {
    std::unique_lock lock(mutex);

    const int firstInFlightBatch = *std::ranges::min_element(inFlightBatches);
    LOG_DEBUG("firstInFlightBatch: {}", firstInFlightBatch);
    currInFlightBatch = firstInFlightBatch;
    lastWaitingBatch = firstInFlightBatch - static_cast<int32_t>(batchSize);
    inFlightBatches.clear();

    if (currInFlightBatch < 0) {
        throw std::runtime_error("Offset cannot be negative.");
    }

    LOG_DEBUG("forgetInFlightBatches");
}

const Dataset &BatchedDataset::getDataset() const noexcept {
    return dataset;
}

const std::atomic_int32_t &BatchedDataset::getLastWaitingBatch() const {
    return lastWaitingBatch;
}

size_t BatchedDataset::getNumberOfBatches() const {
    return (dataset.getDataSource()->getEntries().size() + batchSize - 1) / batchSize;
}
