#include "DecoderRegister.h"
#include "dataDecoders/CompressedDataDecoder.h"
#include "dataDecoders/ExrDataDecoder.h"
#include "dataDecoders/JpgDataDecoder.h"
#include "dataDecoders/NpyDataDecoder.h"
#include "dataDecoders/PngDataDecoder.h"

DecoderRegister &DecoderRegister::getInstance() {
    return *instance;
}

DecoderRegister::DecoderRegister() {
    extToDataDecoder["jpg"] = new JpgDataDecoder(); // TODO: Duplicate code.
    extToDataDecoder["png"] = new PngDataDecoder();
    extToDataDecoder["npy"] = new NpyDataDecoder();
    extToDataDecoder["exr"] = new ExrDataDecoder();
    extToDataDecoder["compressed"] = new CompressedDataDecoder();
}

IDataDecoder *DecoderRegister::getDataDecoderByExtension(const std::string &ext) {
    const auto found = extToDataDecoder.find(ext);
    if (found == extToDataDecoder.end()) return nullptr;
    return found->second;
}
