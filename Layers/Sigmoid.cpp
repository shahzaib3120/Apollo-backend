//
// Created by HP on 11/25/2022.
//

#include "Sigmoid.h"
Sigmoid::Sigmoid() {
    this->inputs = Eigen::MatrixXd::Random(1, 1);
    this->outputs = Eigen::MatrixXd::Random(1, 1);
    this->gradients = Eigen::MatrixXd::Random(1, 1);
}
Sigmoid::Sigmoid(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    this->outputs = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
    this->gradients = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
}
Sigmoid::Sigmoid(int numInputs, int numOutputs) {
    this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->outputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->gradients = Eigen::MatrixXd::Random(numInputs, numOutputs);
}
Sigmoid::Sigmoid(int* shape) {
    this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
    this->outputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
    this->gradients = Eigen::MatrixXd::Random(shape[0], shape[1]);
}
void Sigmoid::setInputs(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
}
Eigen::MatrixXd Sigmoid::getOutputs() {
    return this->outputs;
}
Eigen::MatrixXd Sigmoid::getGradients() {
    return this->gradients;
}
void Sigmoid::setGradients(Eigen::MatrixXd gradients) {
    this->gradients = gradients;
}
void Sigmoid::setOutputs(Eigen::MatrixXd outputs) {
    this->outputs = outputs;
}
void Sigmoid::forward() {
    this->outputs = 1 / (1 + (-this->inputs).array().exp());
}
void Sigmoid::backward() {
    this->gradients = this->outputs.array() * (1 - this->outputs.array());
}
void Sigmoid::forward(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    this->outputs = 1 / (1 + (-this->inputs).array().exp());
}
void Sigmoid::backward(Eigen::MatrixXd gradients) {
    this->gradients = gradients;
    this->gradients = this->outputs.array() * (1 - this->outputs.array());
}
void Sigmoid::update(float learningRate) {
    // dummy function
    return;
}
int* Sigmoid::getInputShape() {
    int* shape = new int[2];
    shape[0] = this->inputs.rows();
    shape[1] = this->inputs.cols();
    return shape;
}
int* Sigmoid::getOutputShape() {
    int* shape = new int[2];
    shape[0] = this->outputs.rows();
    shape[1] = this->outputs.cols();
    return shape;
}

