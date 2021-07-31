'''
Date: 2021-07-23 17:42:36
LastEditors: Liuliang
LastEditTime: 2021-07-23 17:49:56
Description: 
'''
class ListNode():
    def __init__(self, val, next=None) -> None:
        self.val = val
        self.next = next
        pass
a = ListNode(0)
b = ListNode(1)
c = ListNode(2)
a.next = b
b.next = c

# print(a.next.val)

def Printlistre(head):
    if head:
        if head.next:
            Printlistre(head.next)
        print(head.val)

# Printlistre(a)

def pr2(head):
    stack = []
    while head is not None:
        stack.append(head.val)
        head = head.next
    while stack:
        print(stack.pop())

pr2(a)
