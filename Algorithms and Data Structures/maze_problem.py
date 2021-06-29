'''
Date: 2021-06-24 09:02:44
LastEditors: Liuliang
LastEditTime: 2021-06-24 10:17:38
Description: maze
'''



def find_path(x1,y1,x2,y2):
    li    = []
    start = (x1,y1)
    end   = (x2,y2)
    li.append(start)
    while len(li) > 0:
        if li[-1] == end: 
            # print(li)
            return li
        this_node = li[-1]
        for dir in direction:            
            next_node = dir(this_node[0],this_node[1])
            if maze[next_node[0]][next_node[1]] == 0:
                li.append(next_node)
                maze[next_node[0]][next_node[1]] = 2
                break          
        else:
            maze[next_node[0]][next_node[1]] =2
            li.pop()
    return li





if __name__ == '__main__':
    maze=[[1,1,1,1,1,1,1,1,1,1,1,1,1,1],\
          [1,0,0,0,1,1,0,0,0,1,0,0,0,1],\
          [1,0,1,0,0,0,0,1,0,1,0,1,0,1],\
          [1,0,1,0,1,1,1,1,0,1,0,1,0,1],\
          [1,0,1,0,0,0,0,0,0,1,1,1,0,1],\
          [1,0,1,1,1,1,1,1,1,1,0,0,0,1],\
          [1,0,1,0,0,0,0,0,0,0,0,1,0,1],\
          [1,0,0,0,1,1,1,0,1,0,1,1,0,1],\
          [1,0,1,0,1,0,1,0,1,0,1,0,0,1],\
          [1,0,1,0,1,0,1,0,1,1,1,1,0,1],\
          [1,0,1,0,0,0,1,0,0,1,0,0,0,1],\
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
    start=(1,1)
    end=(10,12)

    direction = [
        lambda x, y: (x+1, y),
        lambda x, y: (x-1, y),
        lambda x, y: (x, y+1),
        lambda x, y: (x, y-1)
    ]

    c = find_path(1,1,10,12)
    for i in c:
        print(i)
        
  
