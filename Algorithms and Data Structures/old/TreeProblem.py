'''
Date: 2021-08-06 09:31:19
LastEditors: Liuliang
LastEditTime: 2021-08-06 10:32:38
Description: 
'''

class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None
 
a = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
d = TreeNode(4)
e = TreeNode(5)
f = TreeNode(6)
g = TreeNode(7)
 
a.left = b
a.right = c
b.left = d
b.right = e
c.left = f
c.right = g

#recurrent
def preOrderTraverse_re(node):
    
    if not node:
        return None
    print(node.val)
    preOrderTraverse_re(node.left)
    preOrderTraverse_re(node.right)

# preOrderTraverse(a)

#None recurrent 

def preOrderTravese(node):
    stack = [node]
    while len(stack) > 0:
        print(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
        node = stack.pop()
        
preOrderTravese(a)