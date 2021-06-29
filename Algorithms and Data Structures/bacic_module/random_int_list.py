'''
Date: 2021-06-22 10:02:26
LastEditors: Liuliang
LastEditTime: 2021-06-22 14:32:50
Description: random_int_list
'''

import random

def random_int_list(start,stop,length):
    random_list = []
    start, stop = (int(start),int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    for i in range(length):
        random_list.append(random.randint(start, stop))  
    return random_list


if __name__ == "__main__":

    print(random_int_list(5,1000,20))
    