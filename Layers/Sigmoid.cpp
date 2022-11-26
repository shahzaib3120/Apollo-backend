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
