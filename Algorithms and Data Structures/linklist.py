'''
Date: 2021-06-24 10:56:59
LastEditors: Liuliang
LastEditTime: 2021-06-24 11:26:06
Description: 
'''
class Node():
    def __init__(self,item):
        self.item = item
        self.next = None

a = Node(1)
b = Node(2)
c = Node(3)

a.next = b
b.next = c

print(a.item)
print(a.next.item)
print(b.next.item)
print(c.next.item)
