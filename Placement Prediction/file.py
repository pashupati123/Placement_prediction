import csv
import numpy
import random
columns =5
rows = 50
with open("default.csv", "wb") as outfile:
    writer = csv.writer(outfile)
    for x in range(rows):
        a_list = [random.randint(0,1) for i in range(columns)]
        #a_list = [random.uniform(5, 10) for i in range(columns)]
        writer.writerow(a_list)
