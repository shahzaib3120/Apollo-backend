//
// Created by HP on 12/12/2022.
//

#ifndef APOLLO_LINALG_H
#define APOLLO_LINALG_H
#include <Eigen/Dense>
using namespace std;
namespace Apollo {
    namespace linalg {
        Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, int size, int axis);

        Eigen::MatrixXd broadcast(Eigen::MatrixXd matrix, Eigen::MatrixXd shape, int axis);
    }
}
#endif //APOLLO_LINALG_H
