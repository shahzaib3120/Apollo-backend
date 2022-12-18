//
// Created by HP on 11/26/2022.
//
#include "Tests.h"
#include <iostream>
#include <Eigen/Dense>
#include "../Layers/Sigmoid.h"
#include "../Layers/Dense.h"
#include "../Utils/Dataloader.h"
#include "../Model/Model.h"
#include "../Loss/Loss.h"
#include <highfive/H5Easy.hpp>
//#include <highfive/H5File.hpp>
//using namespace HighFive;
using namespace std;
void testSigmoid() {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(5,1);
    std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
    Sigmoid* sig = new Sigmoid(m);
    sig->forward();
    std::cout << "Here is the matrix sig->getOutputs():" << std::endl << sig->getOutputs() << std::endl;
}
void testDataloader(){
    string path = "E:/Learning-E/Apollo/Dataset/emails-formatted.csv";
    Dataloader dataloader(path);
    dataloader.head(10);
    int* shape = dataloader.getTrainDataShape();
    cout << "Data shape: " << shape[0] << " " << shape[1] << endl;
//    dataloader.showLabels();
}
void testBCE() {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(5,1);
    // take sigmoid of m
    Sigmoid* sig = new Sigmoid(m);
    sig->forward();
    Eigen::MatrixXd m2 = Eigen::MatrixXd::Random(5,1);
    // populate m2 with random 0s and 1s
    for (int i = 0; i < m2.rows(); ++i) {
        for (int j = 0; j < m2.cols(); ++j) {
            m2(i,j) = rand() % 2;
        }
    }
    Eigen::MatrixXd out = sig->getOutputs();
    std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
    std::cout << "Here is sigmoid(m):" << std::endl << sig->getOutputs() << std::endl;
    std::cout << "Here is the matrix m2:" << std::endl << m2 << std::endl;
    auto [grad, loss]= Loss::BCE(out, m2);
    std::cout << "Here is the matrix loss:" << std::endl << grad << std::endl;
    std::cout << "Here is the matrix loss.sum():" << std::endl << loss << std::endl;
    // outputs from python:
    // Original x:  [[-0.997497]
    // [ 0.127171]
    // [-0.613392]
    // [ 0.617481]
    // [ 0.170019]]
    //Output of sigmoid function is:
    // [[0.26943383]
    // [0.53174997]
    // [0.35128583]
    // [0.64964542]
    // [0.54240266]]
    //Lables:
    // [[1]
    // [0]
    // [0]
    // [1]
    // [0]]
    //Loss is:  0.7432085241499561
    //dY_hat is:  [[-3.71148646]
    // [ 2.1356112 ]
    // [ 1.54151095]
    // [-1.53930123]
    // [ 2.18532738]]
}

void testTrain(){
    string path = "E:/Learning-E/Apollo-backend/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo-backend/Dataset/winequality-red.csv";
//    string path = "E:/Learning-E/Apollo-backend/Dataset/data.csv";
//    string path = "F:/Machine-Learning/Apollo-backend/Dataset/data.csv";
    string path = "F:/Machine-Learning/Apollo-backend/Dataset/emails-formatted.csv";
//    string path = "E:/Learning-E/Apollo-backend/Dataset/sampleCircle.csv";
//    string path = "F:/Machine-Learning/Apollo-backend/Dataset/sampleCircle.csv";
    Dataloader dataloader(path, 0.8);
//    dataloader.head(5);
    int* shape = dataloader.getTrainDataShape();
    auto* model =  new Model(shape, true, 0.01, 1);
//    MultiType d1 = Dense(3, shape);
//    model->addLayer(&d1);
//    MultiType s1 = Sigmoid(model->getLastLayerOutputShape());
//    model->addLayer(&s1);
//    MultiType d2 = Dense(1, model->getLastLayerOutputShape());
//    model->addLayer(&d2);
//    MultiType s2 = Sigmoid(model->getLastLayerOutputShape());
//    model->addLayer(&s2);
//    model->compile();
//    model->summary();
//    model->fit(dataloader.getTrainData(), dataloader.getTrainLabels(), dataloader.getValData(), dataloader.getValLabels(), 100, BCE, true);
//    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), BCE);
//    model->saveModel("F:/Machine-Learning/Apollo-backend/Models/spam.csv");
    model->loadModel("F:/Machine-Learning/Apollo-backend/Models/spam.csv");
    model->compile();
    model->summary();
    model->evaluate(dataloader.getValData(), dataloader.getValLabels(), BCE);
}

void testHighFive(){
    string filename = "../Datasets/saved.h5";
//    H5Easy::File file("example.h5", H5Easy::File::Overwrite);
//    Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,2);
//    H5Easy::dump(file, "/path/to/A", m, H5Easy::DumpMode::Overwrite);
    int A = 10;
//    H5Easy::dump(file, "/path/to/A", A, H5Easy::DumpMode::Overwrite);
//    A = H5Easy::load<int>(file, "/path/to/A");
}