//
// Created by HP on 11/25/2022.
//
#include <Eigen/Dense>
#include "Softmax.h"
Apollo::Softmax::Softmax() {
    this->inputs = Eigen::MatrixXd(0,0);
    this->outputs = Eigen::MatrixXd(0,0);
    this->gradients = Eigen::MatrixXd(0,0);
}
Apollo::Softmax::Softmax(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
    this->outputs = Eigen::MatrixXd(0,0);
    this->gradients = Eigen::MatrixXd(0,0);
}
void Apollo::Softmax::setInputs(Eigen::MatrixXd inputs) {
    this->inputs = inputs;
}
Eigen::MatrixXd Apollo::Softmax::getOutputs() {
    return this->outputs;
}
Eigen::MatrixXd Apollo::Softmax::getGradients() {
    return this->gradients;
}
void Apollo::Softmax::setGradients(Eigen::MatrixXd gradients) {
    this->gradients = gradients;
}
void Apollo::Softmax::setOutputs(Eigen::MatrixXd outputs) {
    this->outputs = outputs;
}
//void Apollo::Softmax::forward() {
//    Eigen::MatrixXd exps = this->inputs.array().exp();
//    Eigen::MatrixXd sum = exps.colwise().sum();
//    this->outputs = exps.array().rowwise() / sum.array();
//}
//void Apollo::Softmax::backward() {
//    Eigen::MatrixXd sum = this->gradientsOut.colwise().sum();
//    this->gradientsOut = this->gradientsOut.array().rowwise() - sum.array();
//}
//void Apollo::Softmax::forward(Eigen::MatrixXd inputs) {
//    this->inputs = inputs;
//    Eigen::MatrixXd exps = this->inputs.array().exp();
//    Eigen::MatrixXd sum = exps.colwise().sum();
//    this->outputs = exps.array().rowwise() / sum.array();
//}
//void Apollo::Softmax::backward(Eigen::MatrixXd gradientsOut) {
//    this->gradientsOut = gradientsOut;
//    Eigen::MatrixXd sum = this->gradientsOut.colwise().sum();
//    this->gradientsOut = this->gradientsOut.array().rowwise() - sum.array();
//}

// Todo: check operations on matrices and arrays