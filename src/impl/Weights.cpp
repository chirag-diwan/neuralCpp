#include "../include/Weights.h"

void writeMatrix(std::ofstream& out, const Mat& mat) {
    
    out.write(reinterpret_cast<const char*>(&mat.rows), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&mat.cols), sizeof(uint32_t));
    
    
    out.write(reinterpret_cast<const char*>(mat.data), mat.rows * mat.cols * sizeof(float));
}

void readMatrix(std::ifstream& in, Mat& mat) {
    uint32_t r = 0, c = 0;
    in.read(reinterpret_cast<char*>(&r), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&c), sizeof(uint32_t));

    
    if (r != mat.rows || c != mat.cols) {
        throw std::runtime_error("Binary file architecture mismatch: Expected dimensions do not match pre-allocated matrix.");
    }

    
    in.read(reinterpret_cast<char*>(mat.data), r * c * sizeof(float));
}


void SaveModel(const NeuralNetwork& model, uint32_t inputParamCount, const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open filepath for writing.");
    }
    
    
    uint32_t batchSize = static_cast<uint32_t>(model.batchsize);
    out.write(reinterpret_cast<const char*>(&batchSize),sizeof(uint32_t));
    uint32_t numLayers = static_cast<uint32_t>(model.layerSizes.size());
    out.write(reinterpret_cast<const char*>(&numLayers), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&inputParamCount), sizeof(uint32_t));

    for (size_t size : model.layerSizes) {
        uint32_t s = static_cast<uint32_t>(size);
        out.write(reinterpret_cast<const char*>(&s), sizeof(uint32_t));
    }

    
    for (const auto& layer : model.layers) {
        writeMatrix(out, layer.A);
        writeMatrix(out, layer.W);
        writeMatrix(out, layer.B);
        writeMatrix(out, layer.Z);
        writeMatrix(out, layer.delta);
        writeMatrix(out, layer.zSigmaPrime);
    }
}

NeuralNetwork NNLoadModel(const std::string& filepath , ActivationProfile& Profile) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open filepath for reading.");
    }

    
    uint32_t batchsize;
    uint32_t numLayers = 0;
    uint32_t inputParamCount = 0;
    in.read(reinterpret_cast<char*>(&batchsize),sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&numLayers), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&inputParamCount), sizeof(uint32_t));

    std::vector<size_t> layerSizes(numLayers);
    for (uint32_t i = 0; i < numLayers; i++) {
        uint32_t s = 0;
        in.read(reinterpret_cast<char*>(&s), sizeof(uint32_t));
        layerSizes[i] = static_cast<size_t>(s);
    }

    NeuralNetwork model(std::move(layerSizes));
    model.Init(inputParamCount , batchsize , Profile);

    
    for (auto& layer : model.layers) {
        readMatrix(in, layer.A);
        readMatrix(in, layer.W);
        readMatrix(in, layer.B);
        readMatrix(in, layer.Z);
        readMatrix(in, layer.delta);
        readMatrix(in, layer.zSigmaPrime);
    }

    return model;
}
