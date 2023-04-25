#include <iostream>
#include <fstream>
using namespace std;

double quad(int x)
{
    int a = -2;
    int b = 10;
    int c = 8;

    return (a * x * x + b * x + c);
}

int main()
{
    ofstream file("./quadratic.csv");
    if (file.is_open())
    {
        file << "Pred,x0,x1,x2" << endl;
        for (int i = -10; i <= 10; i++)
        {
            file << quad(i) << "," << 1 << "," << i << "," << i * i << endl;
        }
    }
    return 0;
}