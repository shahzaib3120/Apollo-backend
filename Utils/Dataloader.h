//
// Created by HP on 11/26/2022.
//

#ifndef APOLLO_DATALOADER_H
#define APOLLO_DATALOADER_H

#include <Eigen/Dense>
class Dataloader {
private:
    Eigen::MatrixXd data;
    Eigen::MatrixXd labels;
    int batchSize;
    int numBatches;
    int currentBatch;
    void readData(std::string path);
public:
    Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize);
    Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels);
    // read data from exel files
    // NOTE: Make sure the file contains labels in the first column
    Dataloader(std::string path);
    Eigen::MatrixXd nextBatch();
    Eigen::MatrixXd nextBatch(int batchNumber);
    Eigen::MatrixXd getBatch(int batchNumber);
    Eigen::MatrixXd getBatch();
    Eigen::MatrixXd getLabels();
    Eigen::MatrixXd getLabels(int batchNumber);
    void head(int n);
    void getSize();
    void showLabels();

};


#endif //APOLLO_DATALOADER_H
