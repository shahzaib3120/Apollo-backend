//
// Created by shahr on 18/12/2022.
//

#ifndef APOLLO_PREPROCESSING_H
#define APOLLO_PREPROCESSING_H
#include <Eigen/Dense>
#include <string>
namespace Apollo {
    namespace Preprocessing {
        Eigen::MatrixXd normalize(Eigen::MatrixXd matrix);

        Eigen::MatrixXd standardize(Eigen::MatrixXd matrix);

        Eigen::MatrixXd spamPreprocessingFile(const std::string &path);

        Eigen::MatrixXd spamPreprocessing(const std::string &email);
    }
}
#endif //APOLLO_PREPROCESSING_H
