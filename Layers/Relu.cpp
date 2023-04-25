//
// Created by HP on 4/25/2023.
//

#include "Relu.h"

namespace Apollo {
    Relu::Relu(){
        this->inputs = Eigen::MatrixXd::Random(1, 1);
        this->outputs = Eigen::MatrixXd::Random(1, 1);
        this->gradientsOut = Eigen::MatrixXd::Random(1, 1);
    }
    Relu::Relu(int numInputs, int numOutputs) {
        this->inputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
        this->outputs = Eigen::MatrixXd::Random(numInputs, numOutputs);
        this->gradientsOut = Eigen::MatrixXd::Random(numInputs, numOutputs);
    }
    Relu::Relu(Eigen::MatrixXd &inputs){
        this->inputs = inputs;
        this->outputs = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
        this->gradientsOut = Eigen::MatrixXd::Random(inputs.rows(), inputs.cols());
    }
    Relu::Relu(int *shape){
        this->inputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
        this->outputs = Eigen::MatrixXd::Random(shape[0], shape[1]);
        this->gradientsOut = Eigen::MatrixXd::Random(shape[0], shape[1]);
    }


    void Relu::setInputs(Eigen::MatrixXd &inputs){
        this->inputs = inputs;
    }
    Eigen::MatrixXd Relu::getOutputs(){
        return outputs;
    }
    Eigen::MatrixXd Relu::getGradients() {
        return gradientsOut;
    }
    void Relu::setGradients(Eigen::MatrixXd &gradients){
        this->gradientsOut = gradients;
    }
    void Relu::setOutputs(Eigen::MatrixXd &outputs){
        this->outputs = outputs;
    }

    void Relu::forward() {
        outputs= inputs.cwiseMax(0);
    }
    void Relu::forward(Eigen::MatrixXd &inputs) {
        this->inputs = inputs;
        outputs= inputs.cwiseMax(0);
    }
    void Relu::backward(Eigen::MatrixXd &gradientsIn) {
//        gradientsOut = gradientsIn.cwiseProduct(inputs.cwiseSign());
        gradientsOut = gradientsIn;
    }
    void Relu::update(float learningRate) {
        // dummy function
        return;
    }
    int* Relu::getInputShape() {
        auto* shape = new int[2];
        shape[0] = this->inputs.rows();
        shape[1] = this->inputs.cols();
        return shape;
    }
    int* Relu::getOutputShape() {
        auto* shape = new int[2];
        shape[0] = this->outputs.rows();
        shape[1] = this->outputs.cols();
        return shape;
    }
    void Relu::summary() {
        std::cout << std::left << std::setw(15) << "Relu Layer" << std::setw(20) << "| Input Shape: "<< std::setw(10) <<  std::to_string(this->inputs.rows()) + "x" + std::to_string(this->inputs.cols())  << std::setw(20) << "| Output Shape: " << std::setw(10) << std::to_string(this->outputs.rows()) + "x" + std::to_string(this->outputs.cols())  << std::endl;
    }

} // Apollo