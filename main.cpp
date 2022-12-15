#include <iostream>
#include "Utils/Dataloader.h"
#include "Model/Model.h"
#include <fstream>
using namespace std;
using namespace Eigen;
void saveData(std::string filename, Eigen::MatrixXd matrix) {
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    ofstream file(filename);
    if (file.is_open())
    {
//        file << matrix.format(CSVFormat);
        file << "hello";
        file.close();
    }
}
int main() {
    MatrixXd m = MatrixXd::Random(3, 3);
    saveData("test.csv", m);
    system("pause");
    return 0;
}
