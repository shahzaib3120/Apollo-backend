//
// Created by HP on 11/27/2022.
//

#ifndef APOLLO_LOSS_H
#define APOLLO_LOSS_H
#include <Eigen/Dense>
#include <tuple>
namespace Apollo {
    namespace Loss {
        Eigen::MatrixXd MSE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets);

        std::tuple<Eigen::MatrixXd, float> BCE(Eigen::MatrixXd &outputs, Eigen::MatrixXd &targets);

        float BCEValue(Eigen::MatrixXd &loss);
    }
}
#endif //APOLLO_LOSS_H
