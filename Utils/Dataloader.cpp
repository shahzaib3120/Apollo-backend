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
    // divide data into batches
    this->data = Eigen::MatrixXd::Random(this->numBatches, batchSize);
    this->labels = Eigen::MatrixXd::Random(this->numBatches, batchSize);
    for (int i = 0; i < this->numBatches; i++) {
        for (int j = 0; j < batchSize; j++) {
            this->data(i, j) = data(i * batchSize + j, 0);
            this->labels(i, j) = labels(i * batchSize + j, 0);
        }
    }
    this->batchSize = batchSize;
    this->currentBatch = 0;
}
// for binary classification
Dataloader::Dataloader(Eigen::MatrixXd data, Eigen::MatrixXd labels) {
    this->numBatches = data.rows();
    // flatten data to shape rows*cols, 1
    this->data  = data.reshaped(data.rows()*data.cols(), 1);
    this->labels = labels;
}
// TODO: for multiclass classification

Dataloader::Dataloader(std::string path) {
    // read data from exel file
    fstream file;
    file.open(path, ios::in);
    vector<vector<double>> data;
    vector<vector<int>> labels;
    string line;
    int linesRead = 0;
    while (getline(file, line)) {
        // skip first line
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
    this->labels = Eigen::MatrixXd::Random(1,labels.size());
    this->data = Eigen::MatrixXd::Random(data[0].size(), data.size());
    // store data in Eigen::MatrixXd with shape (data, trainingsamples)
    for (int i = 0; i < data[0].size(); i++) {
        for (int j = 0; j < data.size(); j++) {
            this->data(i, j) = data[j][i];
        }
    }
    // store labels in Eigen::MatrixXd with shape (1, trainingsamples)
    for (int i = 0; i < labels.size(); i++) {
        this->labels(0, i) = labels[i][0];
    }
    this->numBatches = data.size();
    this->batchSize = data[0].size();
    this->currentBatch = 0;
}

void Dataloader::head(int n) {
    // data shape = (features, samples)
    // show first n samples
    for (int i = 0; i < min(data.cols(),n); i++) {
        cout << "Data:" << endl;
        for(int j =0; j<min(data.rows(),n); j++) {
            cout << this->data.col(i)[j] << " ";
        }
        cout << endl;
        cout << "Label:" << endl;
        cout << this->labels.row(0)[i] << endl;
    }
}
int* Dataloader::getDataShape() {
    int* shape = new int[2];
    shape[0] = this->data.rows();
    shape[1] = this->data.cols();
    return shape;
}
int* Dataloader::getLabelsShape() {
    int* shape = new int[2];
    shape[0] = this->labels.rows();
    shape[1] = this->labels.cols();
    return shape;
}
void Dataloader::showLabels() {
    cout << "Labels: " << endl;
    cout << this->labels << endl;
    cout << endl;
}
Eigen::MatrixXd Dataloader::getData() {
    return this->data;
}
Eigen::MatrixXd Dataloader::getLabels() {
    return this->labels;
}

int Dataloader::min(long a, int b) {
    if (a < b) {
            return a;
        }
        return b;
}