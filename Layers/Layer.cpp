//
// Created by HP on 11/25/2022.
//

#include "Layer.h"
// (Neurons, inputShape)
// output : (Neurons, inputShape[1])
// input : (inputShape[0], inputShape[1])
Layer::Layer(int numNeurons, int numInputs, int numOutputs) {
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
    this->weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
    this->biases = Eigen::MatrixXd::Zero(numNeurons, numOutputs);
    this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->outputs = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->gradients = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, numInputs);
    this->biasesGradients = Eigen::MatrixXd::Random(numNeurons, numOutputs);
}
Layer::Layer(int numNeurons, int *shape) {
    this->numNeurons = numNeurons;
    this->numInputs = shape[0];
    this->numOutputs = shape[1];
    this->weights = Eigen::MatrixXd::Random(numNeurons, shape[0]);
    this->biases = Eigen::MatrixXd::Zero(numNeurons, shape[1]);
    this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
    this->outputs = Eigen::MatrixXd::Random(numNeurons, shape[1]);
    this->gradients = Eigen::MatrixXd::Random(numNeurons, shape[1]);
    this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, shape[0]);
    this->biasesGradients = Eigen::MatrixXd::Random(numNeurons, shape[1]);
}
void Layer::setInputs(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
}
Eigen::MatrixXd Layer::getOutputs() {
    return this->outputs;
}
Eigen::MatrixXd Layer::getGradients() {
    return this->gradients;
}
Eigen::MatrixXd Layer::getWeights() {
    return this->weights;
}
Eigen::MatrixXd Layer::getBiases() {
    return this->biases;
}
void Layer::setWeights(Eigen::MatrixXd weights) {
    this->weights = weights;
}
void Layer::setBiases(Eigen::MatrixXd biases) {
    this->biases = biases;
}
void Layer::setGradients(Eigen::MatrixXd gradients) {
    this->gradients = gradients;
}
void Layer::setWeightsGradients(Eigen::MatrixXd weightsGradients) {
    this->weightsGradients = weightsGradients;
}
void Layer::setBiasesGradients(Eigen::MatrixXd biasesGradients) {
    this->biasesGradients = biasesGradients;
}
void Layer::setOutputs(Eigen::MatrixXd outputs) {
    this->outputs = outputs;
}
void Layer::setNumNeurons(int numNeurons) {
    this->numNeurons = numNeurons;
}
void Layer::setNumInputs(int numInputs) {
    this->numInputs = numInputs;
}
void Layer::setNumOutputs(int numOutputs) {
    this->numOutputs = numOutputs;
}
int Layer::getNumNeurons() {
    return this->numNeurons;
}
int Layer::getNumInputs() {
    return this->numInputs;
}
int Layer::getNumOutputs() {
    return this->numOutputs;
}
Eigen::MatrixXd Layer::getWeightsGradients() {
    return this->weightsGradients;
}
Eigen::MatrixXd Layer::getBiasesGradients() {
    return this->biasesGradients;
}
void Layer::initializeWeights() {
    this->weights = Eigen::MatrixXd::Random(this->numNeurons, this->numInputs);
}
void Layer::initializeBiases() {
    this->biases = Eigen::MatrixXd::Zero(this->numNeurons, this->numOutputs);
}
int* Layer::getOutputShape() {
    int* outputShape = new int[2];
    outputShape[0] = this->numNeurons;
    outputShape[1] = this->numOutputs;
    return outputShape;
}
int* Layer::getInputShape() {
    int* inputShape = new int[2];
    inputShape[0] = this->numInputs;
    inputShape[1] = this->numOutputs;
    return inputShape;
}
