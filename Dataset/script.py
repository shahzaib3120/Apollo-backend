import os
import csv
import random


def inCircle(x, y, radius):
    return 1 if x**2 + y**2 <= radius**2 else 0


# write to csv in format (inCircle, x, y)
for i in range(10000):
    # x is random number between -10 and 10
    x = random.random() * 20 - 10
    # y is random number between -10 and 10
    y = random.random() * 20 - 10
    pred = inCircle(x, y, 5)
    with open('data.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([pred, x, y])
