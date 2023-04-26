//
// Created by HP on 11/25/2022.
//

#include "Layer.h"
// (Neurons, inputShape)
// output : (Neurons, inputShape[1])
// input : (inputShape[0], inputShape[1])
namespace Apollo{
    Layer::Layer(){}; // default constructor
    Layer::Layer(std::string name) {
        this->name = name;
    }
    std::string Layer::getName() {
        return this->name;
    }

    // dummy functions
    void Layer::setInputs(Eigen::MatrixXd &inputs) {
        return;
    }
    void Layer::setOutputs(Eigen::MatrixXd &outputs) {
        return;
    }
    void Layer::setGradients(Eigen::MatrixXd &gradients) {
        return;
    }
    Eigen::MatrixXd Layer::getInputs() {
        return Eigen::MatrixXd::Random(1, 1);
    }
    Eigen::MatrixXd Layer::getOutputs() {
        return Eigen::MatrixXd::Random(1, 1);
    }
    Eigen::MatrixXd Layer::getGradients() {
        return Eigen::MatrixXd::Random(1, 1);
    }
    void Layer::forward(Eigen::MatrixXd &inputs) {
        return;
    }
    void Layer::backward(Eigen::MatrixXd &gradientsIn) {
        return;
    }
    void Layer::update(float learningRate) {
        return;
    }
    int* Layer::getInputShape() {
        auto* shape = new int[2];
        shape[0] = 1;
        shape[1] = 1;
        return shape;
    }
    int* Layer::getOutputShape() {
        auto* shape = new int[2];
        shape[0] = 1;
        shape[1] = 1;
        return shape;
    }

    int Layer::getTrainableParams() {
        return 0;
    }
    void Layer::saveWeights(const std::string &path, bool append) {
        return;
    }
    void Layer::saveBiases(const std::string &path, bool append) {
        return;
    }
    void Layer::saveLayer(const std::string &path, bool append) {
        return;
    }
    void Layer::saveGradients(const std::string &path, bool append) {
        return;
    }
}