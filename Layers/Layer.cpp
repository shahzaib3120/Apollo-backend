//
// Created by HP on 11/25/2022.
//

#include "Layer.h"
Layer::Layer(int numNeurons, int numInputs, int numOutputs) {
    this->numNeurons = numNeurons;
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
    this->weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
    this->biases = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->outputs = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->gradients = Eigen::MatrixXd::Random(numNeurons, numOutputs);
    this->weightsGradients = Eigen::MatrixXd::Random(numNeurons, numInputs);
    this->biasesGradients = Eigen::MatrixXd::Random(numNeurons, numOutputs);
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
