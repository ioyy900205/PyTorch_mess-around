'''
Date: 2021-08-12 12:24:42
LastEditors: Liuliang
LastEditTime: 2021-08-12 18:28:35
Description: 
'''
from typing import Optional

class Node():
    def __init__(self, data) -> None:
        self.data = data
        self._next = None

# class LinkList():
#     def __init__(self) -> None:
#         self._head = None

# link = LinkList()
# link._head = Node(1)
# link._head.next = Node(2)


# print(link._head.next.val)

n_1 = Node(1)
n_2 = Node(2)
n_3 = Node(3)
n_4 = Node(4)
n_5 = Node(5)

n_1._next = n_2
n_2._next = n_3
n_3._next = n_4
n_4._next = n_5

def reverse(head: Node) -> Optional[Node]:
    reversed_head = None
    current = head
    while current: 
        # current, reversed_head, reversed_head._next,  = current._next, current, reversed_head
        # current, reversed_head._next, reversed_head,  = current._next, reversed_head, current #这个不行
        reversed_head, reversed_head._next, current  = current, reversed_head,current._next
    return reversed_head

def reverse_cur(head:Node):
    if head == None or head._next == None:
        return head
    else:
        newhead = reverse_cur(head._next)
        head._next._next = head
        head._next = None
        return newhead


def test(head:Node):
    slow, fast = head, head
    while fast and fast._next:
        slow = slow._next
        fast = fast._next._next
        if slow == fast:
            return True
    return False

def merge(l1:Node,l2:Node):
    if l1 and l2:
        p1, p2 = l1, l2
        fake_head = Node(None)
        current = fake_head
        while p1 and p2:
            if p1.data <= p2.data:
                current._next = p1
                p1 = p1._next
            else:
                current._next = p2
                p2 = p2._next
            current = current._next
        current._next = p1 if p1 else p2
        return fake_head._next        
    return l1 or l2

def del_n(head:Node, n:int):
    current = head
    count = 0
    while current is not None:
        count += 1
        current = current._next
    count -= n+1
    current = head
    
    while count>0:
        count -= 1
        current = current._next
    current._next = current._next._next
    return head
    #nums = count - n

def del_n_2(head:Node, n:int):
    fast = head
    count = 0
    while fast and count < n:
        fast = fast._next
        count += 1
    if not fast and count < n:
        return head
    if not fast and count == n:
        return head._next
    
    slow = head
    while fast._next:
        fast, slow = fast._next, slow._next
    slow._next = slow._next._next
    return head
    
    return 0

def print_all(head:Node):
    nums = []
    current = head
    while current:
        nums.append(current.data)
        current = current._next
    print('->'.join(str(num) for num in nums))

# def find_mid(head:Node):
print_all(n_1)
m = reverse(n_1)
print_all(m)
print(test(m))
print(test(n_1))
nums = del_n_2(m,3)
print_all(m)
# print(n_1.data)
# print(n_1._next.data)