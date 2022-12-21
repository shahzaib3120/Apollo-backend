//
// Created by HP on 12/12/2022.
//
#include "linalg.h"

Eigen::MatrixXd Apollo::linalg::broadcast(Eigen::MatrixXd matrix, int size, int axis) {
    Eigen::MatrixXd result;
    if (axis == 0){
        result = Eigen::MatrixXd::Zero(size, matrix.cols());
        for (int i = 0; i < size; i++){
            result.row(i) = matrix;
        }
    }
    else if (axis == 1){
        result = Eigen::MatrixXd::Zero(matrix.rows(), size);
        for (int i = 0; i < size; i++){
            result.col(i) = matrix;
        }
    }
    return result;
}
Eigen::MatrixXd Apollo::linalg::broadcast(Eigen::MatrixXd matrix, Eigen::MatrixXd shape, int axis) {
    Eigen::MatrixXd result;
    if (axis == 0){
        result = Eigen::MatrixXd::Zero(shape(0), matrix.cols());
        for (int i = 0; i < shape(0); i++){
            result.row(i) = matrix;
        }
    }
    else if (axis == 1){
        result = Eigen::MatrixXd::Zero(matrix.rows(), shape(1));
        for (int i = 0; i < shape(1); i++){
            result.col(i) = matrix;
        }
    }
    return result;
}