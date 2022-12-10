//
// Created by HP on 11/27/2022.
//
#include <Eigen/Dense>
#include "Loss.h"
#include <iostream>
Eigen::MatrixXd Loss::MSE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets){
    return (outputs - targets).array().square();
}
std::tuple<Eigen::MatrixXd, float> Loss::BCE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets) {
    // check if shapes are equal
    // print shapes
    if (outputs.rows() != targets.rows() || outputs.cols() != targets.cols()) {
        throw std::invalid_argument("Error: Shape of outputs and targets are not equal");
    }
    // prevent log(0)
    outputs = outputs.array() + 1e-6;
    // calculate BCE
    //  dY_hat = -Y/Y_hat+(1-Y)/(1-Y_hat)
    // loss = -np.sum(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat))/Y.reshape(1,-1).shape[1]
    // TODO : change dY_hat to grads
    Eigen::MatrixXd dY_hat = -targets.array() / outputs.array() + (1 - targets.array()) / (1 - outputs.array());
    float loss = -((targets.array() * outputs.array().log() + (1 - targets.array()) * (1 - outputs.array()).log()).sum()) / targets.size();
    // return tuple of loss and dY_hat
    // print shapes
//    std::cout << "Loss: " << loss << std::endl;
//    std::cout << "dY_hat Shape: " << dY_hat.rows() << "x" << dY_hat.cols() << std::endl;
    return std::make_tuple(dY_hat, loss);
}
