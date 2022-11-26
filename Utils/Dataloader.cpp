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
        // read first 5 lines only
//        if (linesRead > 5) {
//            break;
//        }
        vector<double> row;
        vector<int> label;
        string word;
        int i = 0;
        for (auto x : line) {
            if (x == ',') {
                if (i == 0) {
                    label.push_back(stoi(word));
                } else {
                    row.push_back(stod(word));
                }
                word = "";
                i++;
            } else {
                word = word + x;
            }
        }
        label.push_back(stod(word));
        data.push_back(row);
        labels.push_back(label);
        linesRead++;
    }
    this->data = Eigen::MatrixXd::Random(data.size(), data[0].size());
    this->labels = Eigen::MatrixXd::Random(labels.size(), 1);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            this->data(i, j) = data[i][j];
        }
    }
    for (int i = 0; i < labels.size(); i++) {
        this->labels(i, 0) = labels[i][0];
    }
    this->numBatches = data.size();
    this->batchSize = data[0].size();
    this->currentBatch = 0;
}

void Dataloader::head(int n) {
    cout << "Data:" << endl;
    // show first n rows
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < this->data.cols(); j++) {
            // show first and last 5 columns
            if (j < 5 || j > this->data.cols() - 5) {
                cout << this->data(i, j) << " ";
            }
            // show ... in between
            if (j == 5) {
                cout << "... ";
            }
        }
        // show respective labels
        cout << "\nLabel: " << this->labels(i, 0) << endl;
    }
}
void Dataloader::getSize() {
    cout << "Data size: " << this->data.rows() << "x" << this->data.cols() << endl;
    cout << "Labels size: " << this->labels.rows() << "x" << this->labels.cols() << endl;
}
void Dataloader::showLabels() {
    cout << "Labels: " << endl;
    cout << this->labels << endl;
    cout << endl;
}
