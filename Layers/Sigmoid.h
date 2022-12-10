//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_SIGMOID_H
#define APOLLO_SIGMOID_H
#include <Eigen/Dense>
class Sigmoid{
private:
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd outputs;
    Eigen::MatrixXd gradientsOut;
public:
    Sigmoid();
    Sigmoid(Eigen::MatrixXd &inputs);
    Sigmoid(int numInputs, int numOutputs);
    Sigmoid(int* shape);
    void setInputs(Eigen::MatrixXd inputs);
    Eigen::MatrixXd getOutputs();
    Eigen::MatrixXd getGradients();
    void setGradients(Eigen::MatrixXd &gradients);
    void setOutputs(Eigen::MatrixXd &outputs);
    void forward();
    void backward();
    void update(float learningRate);
    void forward(Eigen::MatrixXd &inputs);
    void backward(Eigen::MatrixXd &gradients);
    int* getInputShape();
    int* getOutputShape();

};
#endif //APOLLO_SIGMOID_H
