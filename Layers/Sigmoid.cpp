//
// Created by HP on 11/25/2022.
//

#include "Sigmoid.h"
namespace Apollo{

    Sigmoid::Sigmoid() {
        this->inputs = Eigen::MatrixXd::Random(1, 1);
        this->outputs = Eigen::MatrixXd::Random(1, 1);
        this->gradientsOut = Eigen::MatrixXd::Random(1, 1);
    }
    Sigmoid::Sigmoid(std::string name, Eigen::MatrixXd &inputs): Layer(name) {
        this->inputs = inputs;
        this->outputs = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
        this->gradientsOut = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
    }
    Sigmoid::Sigmoid(std::string name, int numInputs, int numOutputs): Layer(name) {
        this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
        this->outputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
        this->gradientsOut = Eigen::MatrixXd::Random(numInputs, numOutputs);
    }
    Sigmoid::Sigmoid(std::string name, int* shape): Layer(name) {
        this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
        this->outputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
        this->gradientsOut = Eigen::MatrixXd::Random(shape[0], shape[1]);
    }
    void Sigmoid::setInputs(Eigen::MatrixXd &inputs) {
        this->inputs = inputs;
    }
    void Sigmoid::setOutputs(Eigen::MatrixXd &outputs) {
        this->outputs = outputs;
    }
    void Sigmoid::setGradients(Eigen::MatrixXd &gradients) {
        this->gradientsOut = gradients;
    }
    Eigen::MatrixXd Sigmoid::getInputs() {
        return this->inputs;
    }
    Eigen::MatrixXd Sigmoid::getOutputs() {
        return this->outputs;
    }
    Eigen::MatrixXd Sigmoid::getGradients() {
        return this->gradientsOut;
    }
    void Sigmoid::forward(Eigen::MatrixXd &inputs) {
        this->inputs = inputs;
        this->outputs = 1 / (1 + (-this->inputs).array().exp());
    }
    void Sigmoid::backward(Eigen::MatrixXd &gradientsIn) {
        this->gradientsOut = gradientsIn.array() * this->outputs.array() * (1 - this->outputs.array());
    }
    void Sigmoid::update(float learningRate) {
        // dummy function
        return;
    }
    void Sigmoid::update(float learningRate, float gamma) {
        // dummy function
        return;
    }
    int* Sigmoid::getInputShape() {
        auto* shape = new int[2];
        shape[0] = this->inputs.rows();
        shape[1] = this->inputs.cols();
        return shape;
    }
    int* Sigmoid::getOutputShape() {
        auto* shape = new int[2];
        shape[0] = this->outputs.rows();
        shape[1] = this->outputs.cols();
        return shape;
    }
    void Sigmoid::summary() {
        cout << left << setw(15) << "Sigmoid Layer" << setw(20) << "| Input Shape: "<< setw(10) <<  to_string(this->inputs.rows()) + "x" + to_string(this->inputs.cols())  << setw(20) << "| Output Shape: " << setw(10) << to_string(this->outputs.rows()) + "x" + to_string(this->outputs.cols())  << endl;
    }

}
