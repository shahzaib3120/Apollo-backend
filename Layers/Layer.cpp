//
// Created by HP on 11/25/2022.
//

#include "Layer.h"
// (Neurons, inputShape)
// output : (Neurons, inputShape[1])
// input : (inputShape[0], inputShape[1])
Apollo::Layer::Layer(int numNeurons, int numInputs, int numOutputs) {
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
    this->weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
    this->biases = Eigen::VectorXd::Zero(numNeurons);
    this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->outputs = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->gradients = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, numInputs);
    this->biasesGradients = Eigen::VectorXd::Random(numNeurons);
}
Apollo::Layer::Layer(int numNeurons, int *shape) {
    this->numNeurons = numNeurons;
    this->numInputs = shape[0];
    this->numOutputs = shape[1];
    this->weights = Eigen::MatrixXd::Random(numNeurons, shape[0]);
    this->biases = Eigen::VectorXd::Zero(numNeurons);
    this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
    this->outputs = Eigen::MatrixXd::Random(numNeurons, shape[1]);
    this->gradients = Eigen::MatrixXd::Random(numNeurons, shape[1]);
    this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, shape[0]);
    this->biasesGradients = Eigen::VectorXd::Random(numNeurons);
}

Apollo::Layer::Layer(Eigen::MatrixXd weights, Eigen::VectorXd biases, int numOutputs) {
    this->weights = weights;
    this->biases = biases;
    this->numNeurons = weights.rows();
    this->numInputs = weights.cols();
    this->numOutputs = numOutputs;
    this->inputs = Eigen::MatrixXd::Random(weights.cols(), numOutputs);
    this->outputs = Eigen::MatrixXd::Random(weights.rows(), numOutputs);
    this->gradients = Eigen::MatrixXd::Random(weights.rows(), numOutputs);
    this->weightsGradients = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
    this->biasesGradients = Eigen::VectorXd::Random(weights.rows());
}

void Apollo::Layer::setInputs(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
}
Eigen::MatrixXd Apollo::Layer::getOutputs() {
    return this->outputs;
}
Eigen::MatrixXd Apollo::Layer::getGradients() {
    return this->gradients;
}
Eigen::MatrixXd Apollo::Layer::getWeights() {
    return this->weights;
}
Eigen::VectorXd Apollo::Layer::getBiases() {
    return this->biases;
}
void Apollo::Layer::setWeights(Eigen::MatrixXd weights) {
    this->weights = weights;
}
void Apollo::Layer::setBiases(Eigen::VectorXd biases) {
    this->biases = biases;
}
void Apollo::Layer::setGradients(Eigen::MatrixXd gradients) {
    this->gradients = gradients;
}
void Apollo::Layer::setWeightsGradients(Eigen::MatrixXd weightsGradients) {
    this->weightsGradients = weightsGradients;
}
void Apollo::Layer::setBiasesGradients(Eigen::MatrixXd biasesGradients) {
    this->biasesGradients = biasesGradients;
}
void Apollo::Layer::setOutputs(Eigen::MatrixXd outputs) {
    this->outputs = outputs;
}
void Apollo::Layer::setNumNeurons(int numNeurons) {
    this->numNeurons = numNeurons;
}
void Apollo::Layer::setNumInputs(int numInputs) {
    this->numInputs = numInputs;
}
void Apollo::Layer::setNumOutputs(int numOutputs) {
    this->numOutputs = numOutputs;
}
int Apollo::Layer::getNumNeurons() {
    return this->numNeurons;
}
int Apollo::Layer::getNumInputs() {
    return this->numInputs;
}
int Apollo::Layer::getNumOutputs() {
    return this->numOutputs;
}
Eigen::MatrixXd Apollo::Layer::getWeightsGradients() {
    return this->weightsGradients;
}
Eigen::VectorXd Apollo::Layer::getBiasesGradients() {
    return this->biasesGradients;
}
void Apollo::Layer::initializeWeights() {
    this->weights = Eigen::MatrixXd::Random(this->numNeurons, this->numInputs);
}
void Apollo::Layer::initializeBiases() {
    this->biases = Eigen::VectorXd::Zero(this->numNeurons);
}
int* Apollo::Layer::getOutputShape() {
    int* outputShape = new int[2];
    outputShape[0] = this->numNeurons;
    outputShape[1] = this->numOutputs;
    return outputShape;
}
int* Apollo::Layer::getInputShape() {
    int* inputShape = new int[2];
    inputShape[0] = this->numInputs;
    inputShape[1] = this->numOutputs;
    return inputShape;
}

void Apollo::Layer::saveBiases(const std::string &path, bool append) {
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    if (append) {
        file.open(path, std::ios_base::app);
    } else {
        file.open(path);
    }
    // add new line if appending
    if (append) {
        file << "\n";
    }
    // write shape
//    file << this->numNeurons << "," << "1" << "\n";
    // write biases
    file << this->biases.format(CSVFormat);
    file << "\nend";
    file.close();
}

void Apollo::Layer::saveWeights(const std::string &path, bool append) {
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    if (append) {
        file.open(path, std::ios_base::app);
    } else {
        file.open(path);
    }
    // add new line if appending
    if (append) {
        file << "\n";
    }
    // write shapes
//    file << this->numNeurons << "," << this->numInputs << "\n";
    // write weights
    file << this->weights.format(CSVFormat);
    file << "\nend";
    file.close();
}

void Apollo::Layer::saveLayer(const std::string &path, bool append) {
    this->saveBiases(path, append);
    this->saveWeights(path, true);
}
