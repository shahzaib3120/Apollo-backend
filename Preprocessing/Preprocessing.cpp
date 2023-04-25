//
// Created by shahr on 18/12/2022.
//
#include "Preprocessing.h"
#include <unordered_map>
#include <fstream>
using namespace std;
Eigen::MatrixXd Apollo::Preprocessing::spamPreprocessingFile(const std::string &path) {
    // Args:
    // path: path to txt file containing email
    // Returns:
    // Eigen::MatrixXd: matrix
    // Description:
    // This function checks the frequency of predefined wordsMap in the email and returns a matrix containing the frequency of each word
    // in the email.
    // Example:
    // spamPreprocessing("email.txt")
    // Output:
    // [1,10,6,7]
    // predefined wordsMap are 3000 wordsMap stored in a csv file
    // single row containing all the wordsMap separated by commas
//    string wordsPath = "files/words.csv";
//    string wordsPath = "F:/Machine-Learning/Apollo-backend/Preprocessing/files/words.csv";
    string wordsPath = "E:/Learning-E/Apollo-backend/Preprocessing/files/words.csv";
//    string wordsPath = "F:/Machine-Learning/Apollo-backend/Preprocessing/files/demoWords.csv";
    // create a dictionary of wordsMap and set their frequency to 0

    unordered_map<string, int> wordsMap;
    vector<string> insertionOrder;
    ifstream wordsFile(wordsPath);
    string word;
    while (getline(wordsFile, word, ',')) {
        wordsMap[word] = 0;
        insertionOrder.push_back(word);
    }
    // read the email and update the frequency of each word
    ifstream emailFile(path);
    // remove punctuation and convert to lowercase
    string emailLine;
    string email;
    while (getline(emailFile, emailLine)) {
        for (int i = 0; i < emailLine.length(); i++) {
            if (ispunct(emailLine[i])) {
                emailLine.erase(i--, 1);
            }
            emailLine[i] = tolower(emailLine[i]);
        }
        email += emailLine;
    }
    word = "";
    istringstream iss(email);
    while (iss >> word) {
        if (wordsMap.find(word) != wordsMap.end()) {
            wordsMap[word]++;
        }
    }
    // create a matrix containing the frequency of each word
    // shape is (wordsMap , 1)
    Eigen::MatrixXd matrix(wordsMap.size(), 1);
    int i = 0;
    for (auto &it : insertionOrder) {
        matrix(i, 0) = wordsMap[it];
        i++;
    }
    return matrix;
}

Eigen::MatrixXd Apollo::Preprocessing::spamPreprocessing(const std::string &email) {
    // Args:
    // email: string containing email
    // Returns:
    // Eigen::MatrixXd: matrix
    // Description:
    // This function checks the frequency of predefined wordsMap in the email and returns a matrix containing the frequency of each word
    // in the email.
    // Example:
    // spamPreprocessing("email.txt")
    // Output:
    // [1,10,6,7]
    // predefined wordsMap are 3000 wordsMap stored in a csv file
    // single row containing all the wordsMap separated by commas

    string wordsPath = "E:/Learning-E/Apollo-backend/Preprocessing/files/words.csv";
//    string wordsPath = "files/wordsMap.csv";
    // create a dictionary of wordsMap and set their frequency to 0
    unordered_map<string, int> wordsMap;
    vector<string> insertionOrder;
    ifstream wordsFile(wordsPath);
    string word;
    while (getline(wordsFile, word, ',')) {
        wordsMap[word] = 0;
        insertionOrder.push_back(word);
    }
    word = "";
    // read the email and update the frequency of each word
    // remove punctuation and convert to lowercase
    string email2 = email;
    // remove new line characters
    email2.erase(remove(email2.begin(), email2.end(), '\n'), email2.end());
    for (int i = 0; i < email2.length(); i++) {
        if (ispunct(email2[i])) {
            email2.erase(i--, 1);
        }
        email2[i] = tolower(email2[i]);
    }
    // get word in string email 2 and update frequency in wordsMap map
    istringstream iss(email2);
    while (iss >> word) {
        if (wordsMap.find(word) != wordsMap.end()) {
            wordsMap[word]++;
        }
    }
    // create a matrix containing the frequency of each word
    // shape is (wordsMap , 1)
    Eigen::MatrixXd matrix(wordsMap.size(), 1);
    int i = 0;
    for (auto &it : insertionOrder) {
        matrix(i, 0) = wordsMap[it];
        i++;
    }
    return matrix;
}

Eigen::MatrixXd Apollo::Preprocessing::normalize(Eigen::MatrixXd matrix) {
    // Args:
    // matrix: matrix to be normalized
    // Returns:
    // Eigen::MatrixXd: normalized matrix
    // Description:
    // This function normalizes the matrix by dividing each element by the maximum element in the matrix
    // Example:
    // normalize([[1,2,3],[4,5,6]])
    // Output:
    // [[0.16666666666666666, 0.3333333333333333, 0.5],
    // [0.6666666666666666, 0.8333333333333334, 1.0]]
    double max = matrix.maxCoeff();
    return matrix / max;
}

Eigen::MatrixXd Apollo::Preprocessing::standardize(Eigen::MatrixXd matrix) {
    // Args:
    // matrix: matrix to be standardized
    // Returns:
    // Eigen::MatrixXd: standardized matrix
    // Description:
    // This function standardizes the matrix by subtracting the mean and dividing by the standard deviation
    // Example:
    // standardize([[1,2,3],[4,5,6]])
    // Output:
    // [[-1.224744871391589, -0.4082482904638631, 0.4082482904638631],
    // [1.224744871391589, 0.4082482904638631, -0.4082482904638631]]
    // calculate the mean of the matrix
    double mean = matrix.mean();
    // compute the standard deviation
    double std = 0;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            std += pow(matrix(i, j) - mean, 2);
        }
    }
    std = sqrt(std / (matrix.rows() * matrix.cols()));
    // subtract the mean and divide by the standard deviation
    Eigen::MatrixXd standardizedMatrix = matrix;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            standardizedMatrix(i, j) = (matrix(i, j) - mean) / std;
        }
    }
    return standardizedMatrix;
}