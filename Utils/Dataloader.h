//
// Created by HP on 11/26/2022.
//

#ifndef APOLLO_DATALOADER_H
#define APOLLO_DATALOADER_H

#include <Eigen/Dense>
namespace Apollo{
    class Dataloader {
    private:
        Eigen::MatrixXd trainData;
        Eigen::MatrixXd trainLabels;
        Eigen::MatrixXd valData;
        Eigen::MatrixXd valLabels;
        int batchSize;
        int numBatches;
        int currentBatch;
        static int min(long a, int b);
    public:
        Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize);
        Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels);
        // read trainData from exel files
        // NOTE: Make sure the file contains trainLabels in the first column
        Dataloader(std::string const &path);
        // dataloader for csv files with train and validation split
        Dataloader(std::string const &path, float trainSplit);

        Eigen::MatrixXd &getTrainLabels();
        Eigen::MatrixXd &getTrainData();
        Eigen::MatrixXd &getValData();
        Eigen::MatrixXd &getValLabels();
        
        void head(int n);
        int* getTrainDataShape();
        int* getTrainLabelsShape();
        int * getValDataShape();
        int* getValLabelsShape();

    };
}

#endif //APOLLO_DATALOADER_H
