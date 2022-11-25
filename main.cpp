#include <iostream>
#include <Eigen/Dense>
using namespace std;
int main() {
    Eigen::Matrix3d m = Eigen::Matrix3d::Random();
    std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
    return 0;
}
