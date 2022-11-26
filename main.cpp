#include <iostream>
#include <Eigen/Dense>
#include "Layers/Sigmoid.h"
#include "Utils/Dataloader.h"
using namespace std;
int main() {
    string path = "E:/Learning-E/Apollo/Dataset/emails-formatted.csv";
    Dataloader dataloader(path);
    dataloader.head(10);
    dataloader.getSize();
//    dataloader.showLabels();
    return 0;
}
