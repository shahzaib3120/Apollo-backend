//
// Created by HP on 11/25/2022.
//

#include "Sigmoid.h"
Apollo::Sigmoid::Sigmoid() {
    this->inputs = Eigen::MatrixXd::Random(1, 1);
    this->outputs = Eigen::MatrixXd::Random(1, 1);
    this->gradientsOut = Eigen::MatrixXd::Random(1, 1);
}
Apollo::Sigmoid::Sigmoid(Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    this->outputs = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
    this->gradientsOut = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
}
Apollo::Sigmoid::Sigmoid(int numInputs, int numOutputs) {
    this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->outputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
    this->gradientsOut = Eigen::MatrixXd::Random(numInputs, numOutputs);
}
Apollo::Sigmoid::Sigmoid(int* shape) {
    this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
    this->outputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
    this->gradientsOut = Eigen::MatrixXd::Random(shape[0], shape[1]);
}
void Apollo::Sigmoid::setInputs(Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
}
Eigen::MatrixXd Apollo::Sigmoid::getOutputs() {
    return this->outputs;
}
Eigen::MatrixXd Apollo::Sigmoid::getGradients() {
    return this->gradientsOut;
}
void Apollo::Sigmoid::setGradients(Eigen::MatrixXd &gradients) {
    this->gradientsOut = gradients;
}
void Apollo::Sigmoid::setOutputs(Eigen::MatrixXd &outputs) {
    this->outputs = outputs;
}
void Apollo::Sigmoid::forward() {
    this->outputs = 1 / (1 + (-this->inputs).array().exp());
}
void Apollo::Sigmoid::forward(Eigen::MatrixXd &inputs) {
    this->inputs = inputs;
    this->outputs = 1 / (1 + (-this->inputs).array().exp());
}
void Apollo::Sigmoid::backward(Eigen::MatrixXd &gradientsIn) {
    this->gradientsOut = gradientsIn.array() * this->outputs.array() * (1 - this->outputs.array());
}
void Apollo::Sigmoid::update(float learningRate) {
    // dummy function
    return;
}
int* Apollo::Sigmoid::getInputShape() {
    auto* shape = new int[2];
    shape[0] = this->inputs.rows();
    shape[1] = this->inputs.cols();
    return shape;
}
int* Apollo::Sigmoid::getOutputShape() {
    auto* shape = new int[2];
    shape[0] = this->outputs.rows();
    shape[1] = this->outputs.cols();
    return shape;
}
void Apollo::Sigmoid::summary() {
    cout << left << setw(15) << "Sigmoid Layer" << setw(20) << "| Input Shape: "<< setw(10) <<  to_string(this->inputs.rows()) + "x" + to_string(this->inputs.cols())  << setw(20) << "| Output Shape: " << setw(10) << to_string(this->outputs.rows()) + "x" + to_string(this->outputs.cols())  << endl;
}

