//
// Created by HP on 11/26/2022.
//

#include "Dataloader.h"
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
Dataloader::Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels, int batchSize) {
    this->numBatches = data.rows() / batchSize;
    // divide trainData into batches
    this->trainData = Eigen::MatrixXd::Random(this->numBatches, batchSize);
    this->trainLabels = Eigen::MatrixXd::Random(this->numBatches, batchSize);
    for (int i = 0; i < this->numBatches; i++) {
        for (int j = 0; j < batchSize; j++) {
            this->trainData(i, j) = data(i * batchSize + j, 0);
            this->trainLabels(i, j) = labels(i * batchSize + j, 0);
        }
    }
    this->batchSize = batchSize;
    this->currentBatch = 0;
}
// for binary classification
Dataloader::Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels) {
    this->numBatches = data.rows();
    // flatten trainData to shape rows*cols, 1
    this->trainData  = data.reshaped(data.rows() * data.cols(), 1);
    this->trainLabels = labels;
}
// TODO: for multiclass classification

Dataloader::Dataloader(std::string const path) {
    // Args:
    //     path: path to the csv file
    // Format:
    //     first column is the label
    //     the first row is the header
    //     the rest of the columns are the features

    fstream file;
    file.open(path, ios::in);
    if (!file.is_open()) {
        throw invalid_argument("File not found");
    }
    vector<vector<double>> data;
    vector<vector<int>> labels;
    string line;
    int linesRead = 0;
    while (getline(file, line)) {
        // skip first line (header)
        if (linesRead == 0) {
            linesRead++;
            continue;
        }
        vector<double> row;
        vector<int> label;
        string word;
        int i = 0;
        stringstream s(line);
        while (getline(s, word, ',')) {
            if (i == 0) {
                label.push_back(stoi(word));
            } else {
                row.push_back(stod(word));
            }
            i++;
        }
        label.push_back(stod(word));
        data.push_back(row);
        labels.push_back(label);
        linesRead++;
    }
    this->trainLabels = Eigen::MatrixXd::Random(1, labels.size());
    this->trainData = Eigen::MatrixXd::Random(data[0].size(), data.size());
    // store trainData in Eigen::MatrixXd with shape (trainData, trainingsamples)
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            this->trainData(i, j) = data[j][i];
        }
    }
    // store trainLabels in Eigen::MatrixXd with shape (1, trainingsamples)
    for (int i = 0; i < labels.size(); i++) {
        this->trainLabels(0, i) = labels[i][0];
    }
    this->numBatches = data.size();
    this->batchSize = data[0].size();
    this->currentBatch = 0;
}

Dataloader::Dataloader(std::string path, float trainSplit) {
    // Args:
    //     path: path to the csv file
    //     trainSplit: percentage of trainData to be used for training
    // Format:
    //     first column is the label
    //     the first row is the header
    //     the rest of the columns are the features

    fstream file;
    file.open(path, ios::in);
    if (!file.is_open()) {
        throw invalid_argument("File not found");
    }
    vector<vector<double>> data;
    vector<vector<int>> labels;
    string line;
    int linesRead = 0;
    while (getline(file, line)) {
        // skip first line (header)
        if (linesRead == 0) {
            linesRead++;
            continue;
        }
        vector<double> row;
        vector<int> label;
        string word;
        int i = 0;
        stringstream s(line);
        while (getline(s, word, ',')) {
            if (i == 0) {
                label.push_back(stoi(word));
            } else {
                row.push_back(stod(word));
            }
            i++;
        }
        label.push_back(stod(word));
        data.push_back(row);
        labels.push_back(label);
        linesRead++;
    }
    int trainSize = (int) (data.size() * trainSplit);
    int valSize = data.size() - trainSize;
    this->trainLabels = Eigen::MatrixXd::Random(1, trainSize);
    this->trainData = Eigen::MatrixXd::Random(data[0].size(), trainSize);
    // store trainData in Eigen::MatrixXd with shape (trainData, trainingsamples)
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < trainSize; j++) {
            this->trainData(i, j) = data[j][i];
        }
    }
    // store trainLabels in Eigen::MatrixXd with shape (1, trainingsamples)
    for (int i = 0; i < trainSize; i++) {
        this->trainLabels(0, i) = labels[i][0];
    }
    this->numBatches = trainSize;
    this->batchSize = data[0].size();
    this->currentBatch = 0;
    this->valData = Eigen::MatrixXd::Random(data[0].size(), valSize);
    this->valLabels = Eigen::MatrixXd::Random(1, valSize);
    // store valData in Eigen::MatrixXd with shape (trainData, trainingsamples)
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < valSize; j++) {
            this->valData(i, j) = data[trainSize + j][i];
        }
    }
    // store valLabels in Eigen::MatrixXd with shape (1, trainingsamples)
    for (int i = 0; i < valSize; i++) {
        this->valLabels(0, i) = labels[trainSize + i][0];
    }

}

void Dataloader::head(int n) {

    // Args:
    //     n: number of rows to print

    // trainData shape = (features, samples)
    // show first n samples
    for (int i = 0; i < min(trainData.cols(), n); i++) {
        cout << "Data:" << endl;
        // show first n features
        for(int j =0; j<min(trainData.rows(), n); j++) {
            cout << this->trainData.col(i)[j] << " ";
        }
        cout << endl;
        cout << "Label:" << endl;
        cout << this->trainLabels.row(0)[i] << endl;
    }
}
int* Dataloader::getTrainDataShape() {
    auto* shape = new int[2];
    shape[0] = this->trainData.rows();
    shape[1] = this->trainData.cols();
    return shape;
}
int* Dataloader::getTrainLabelsShape() {
    auto* shape = new int[2];
    shape[0] = this->trainLabels.rows();
    shape[1] = this->trainLabels.cols();
    return shape;
}
int* Dataloader::getValDataShape() {
    auto* shape = new int[2];
    shape[0] = this->valData.rows();
    shape[1] = this->valData.cols();
    return shape;
}
int* Dataloader::getValLabelsShape() {
    auto* shape = new int[2];
    shape[0] = this->valLabels.rows();
    shape[1] = this->valLabels.cols();
    return shape;
}
Eigen::MatrixXd Dataloader::getTrainData() {
    return this->trainData;
}
Eigen::MatrixXd Dataloader::getTrainLabels() {
    return this->trainLabels;
}
Eigen::MatrixXd Dataloader::getValData() {
    return this->valData;
}
Eigen::MatrixXd Dataloader::getValLabels() {
    return this->valLabels;
}

int Dataloader::min(long a, int b) {
    if (a < b) {
            return a;
        }
        return b;
}