'''
Date: 2021-08-10 09:39:50
LastEditors: Liuliang
LastEditTime: 2021-08-11 10:31:24
Description: 
'''
# ================================================================== #
#                说明：1.定义节点                                             
# ================================================================== #	
class Node():
    def __init__(self,val) -> None:
        self.val = val
        self.next = None
        pass
    def __repr__(self) -> str:
        return self.val

#测试                                             

st_1 = Node('1')
st_2 = Node('2')
st_3 = Node('3')
st_1.next = st_2
st_2.next = st_3

# ================================================================== #
#                说明：2.定义链表                                             
# ================================================================== #	
class SingleLinkList():
    def __init__(self) -> None:
        self._head = None

#test
test_1 = 0
if test_1:
    
    link_list = SingleLinkList()
    node1 = Node(1)
    node2 = Node(2)

    link_list._head = node1
    node1.next = node2
    print(link_list._head.val)
    print(link_list._head.next.val)


# ================================================================== #
#                说明：
# 是不是感觉很麻烦，所以我们要在链表中增加操作方法。
# is_empty() 链表是否为空
# length() 链表长度
# items() 获取链表数据迭代器
# add(item) 链表头部添加元素
# append(item) 链表尾部添加元素
# insert(pos, item) 指定位置添加元素
# remove(item) 删除节点
# find(item) 查找节点是否存在                                             
# ================================================================== #



class SingleLinkList_2():
    def __init__(self) -> None:
        self._head = None
    
    def is_empty(self):
        return self._head is None
    
    def length(self):
        cur = self._head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def items(self):
        cur = self._head
        while cur is not None:
            yield cur.val
            cur = cur.next

    def add(self,val):
        node = Node(val)
        node.next = self._head
        self._head = node

    def append(self,val):
        node = Node(val)
        if self.is_empty():
            self._head = node
        else:
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

            
    

link_list_2 = SingleLinkList_2()

# print(link_list_2.is_empty())
# print(link_list_2.length())
# print(link_list_2.items())
link_list_2.add(1)
link_list_2.add(2)
link_list_2.append(3)
# print(link_list_2.is_empty())
# print(link_list_2.length())

print(link_list_2._head.val)
print(link_list_2._head.next.val)
print(link_list_2._head.next.next.val)

