//
// Created by HP on 11/27/2022.
//
#include <Eigen/Dense>
#include "Loss.h"
#include <iostream>
namespace Apollo{
    namespace Loss{
        std::tuple<Eigen::MatrixXd, float> MSE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets){
            /*  Args:
                    outputs: output of the model
                    targets: target values
                Returns:
                    dY_hat: derivative of Y_hat
                    loss: loss value
            */
            // check if shapes are equal
            if (outputs.rows() != targets.rows() || outputs.cols() != targets.cols()) {
                throw std::invalid_argument("Error: Shape of outputs and targets are not equal");
            }
            // calculate MSE
            //  dY_hat = (Y_hat-Y)/Y.reshape(1,-1).shape[1]
            // loss = np.sum((Y_hat-Y)**2)/Y.reshape(1,-1).shape[1]

            Eigen::MatrixXd dY_hat = (outputs-targets)/targets.size();
            float loss = ((outputs-targets).array().pow(2).sum())/targets.size();
            // return tuple of loss and dY_hat
            return std::make_tuple(dY_hat, loss);
        }

        std::tuple<Eigen::MatrixXd, float> BCE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets) {
            /*  Args:
                    outputs: output of the model
                    targets: target values
                Returns:
                    dY_hat: derivative of Y_hat
                    loss: loss value
            */
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
            return std::make_tuple(dY_hat, loss);
        }
    }
}
