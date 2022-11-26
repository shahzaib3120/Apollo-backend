//
// Created by HP on 11/26/2022.
//
#include "Tests.h"
#include <iostream>
#include <Eigen/Dense>
#include "../Layers/Sigmoid.h"
#include "../Layers/Dense.h"
#include "../Layers/Softmax.h"
#include "../Utils/Dataloader.h"
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
    dataloader.getSize();
//    dataloader.showLabels();
}