#include "dataio.h"
#include <utility>
#include <vector>
#include <filesystem>
#include <format>
#include <utils.h>

#include "dataDecoders/CompressedDataDecoder.h"
#include "dataDecoders/ExrDataDecoder.h"
#include "dataDecoders/JpgDataDecoder.h"
#include "dataDecoders/NpyDataDecoder.h"
#include "dataDecoders/PngDataDecoder.h"

Dataset::Dataset(IDataSource *_dataSource,
                 std::vector<IDataTransformAugmentation<2> *> _dataAugmentations,
                 const pybind11::function &createDatasetFunction,
                 const bool isVirtualDataset
) : dataSource(_dataSource), dataAugmentations(std::move(_dataAugmentations)) {
    extToDataDecoder["jpg"] = new JpgDataDecoder();
    extToDataDecoder["png"] = new PngDataDecoder();
    extToDataDecoder["npy"] = new NpyDataDecoder();
    extToDataDecoder["exr"] = new ExrDataDecoder();
    extToDataDecoder["compressed"] = new CompressedDataDecoder();

    if (dataSource->preInitDataset(existsEnvVar(INVALID_DS_ENV_VAR) && isVirtualDataset)) {
        createDatasetFunction();
    }

    dataSource->initDataset();
}

std::tuple<Dataset, Dataset, Dataset> Dataset::splitTrainValidationTest(
    const float trainPercentage, const float validPercentage) {
    if (trainPercentage <= 0.0f || validPercentage <= 0.0f) {
        throw std::runtime_error(
            "Train and validation set must contain more than 0% of elements.");
    }

    const int numTrain = static_cast<int>(std::round(trainPercentage * static_cast<float>(entries.size())));
    const int numValid = static_cast<int>(std::round(validPercentage * static_cast<float>(entries.size())));

    if (numTrain + numValid > static_cast<int>(entries.size())) {
        throw std::runtime_error(
            "Violated #train examples + #validation examples <= #all examples.");
    }

    const std::vector trainEntries(entries.begin(), entries.begin() + numTrain);
    const std::vector validEntries(entries.begin() + numTrain, entries.begin() + numTrain + numValid);
    const std::vector testEntries(entries.begin() + numTrain + numValid, entries.end());

    return std::make_tuple<>(
        Dataset(rootDir + "", std::vector(heads), trainEntries),
        Dataset(rootDir + "", std::vector(heads), validEntries),
        Dataset(rootDir + "", std::vector(heads), testEntries)
    );
}

IDataSource *Dataset::getDataSource() const {
    return dataSource;
}

IDataDecoder *getDataDecoderByExtension(const std::string &ext) const {

}

std::vector<IDataTransformAugmentation<2> *> Dataset::getDataTransformAugmentations() const {
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

        for (const auto& subPath: entries[i % entries.size()]) {
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
