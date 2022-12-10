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