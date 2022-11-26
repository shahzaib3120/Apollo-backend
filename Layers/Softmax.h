//
// Created by HP on 11/25/2022.
//

#ifndef APOLLO_SOFTMAX_H
#define APOLLO_SOFTMAX_H

#include <Eigen/Dense>
class Softmax {
private:
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd outputs;
    Eigen::MatrixXd gradients;
public:
    Softmax();
    Softmax(Eigen::MatrixXd inputs);
    void setInputs(Eigen::MatrixXd inputs);
    Eigen::MatrixXd getOutputs();
    Eigen::MatrixXd getGradients();
    void setGradients(Eigen::MatrixXd gradients);
    void setOutputs(Eigen::MatrixXd outputs);
    void forward();
    void backward();
    void forward(Eigen::MatrixXd inputs);
    void backward(Eigen::MatrixXd gradients);
};


#endif //APOLLO_SOFTMAX_H
