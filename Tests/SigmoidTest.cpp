//
// Created by HP on 11/26/2022.
//
#include <iostream>
#include "../Layers/Sigmoid.h"
using namespace std;
int main() {
    Eigen::Matrix3d m = Eigen::Matrix3d::Random();
    std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
    Sigmoid* sig = new Sigmoid(m);
    cout << arr[1]->getOutputs();
    return 0;
}