'''
Date: 2021-07-12 12:19:09
LastEditors: Liuliang
LastEditTime: 2021-07-12 15:23:28
Description: backtracing_word
'''


def exist(board, word):
    road = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ] #direction
    m = len(board)
    n = len(board[0])


    def dfs(x, y, word):
        res = False
        if len(word) == 1 and board[x][y] == word[0]:
            return True
        elif board[x][y] != word[0]:
            return False
            
        board[x][y] = '*'
        
        for i in range(len(road)):
            newx = x + road[i][0]
            newy = y + road[i][1]
            
            if newx >= m or newx<0 or newy >= n or newy < 0:
                continue
            if board[newx][newy] == '*':
                continue
            res = dfs(newx, newy, word[1:])
            
            if res: return True
            
        if res == False:
            board[x][y] = word[0]
        return False

    for i in range(m):
        for j in range(n):
            if board[i][j]==word[0]:
                res = dfs(i,j,word)
                if res: return True
    return False
    


if __name__ =='__main__':
    board = [
    ['A','B','C','E'],
    ['S','F','C','S'],
    ['A','D','E','E']
    ]
    word = 'SEE'
    out = exist(board, word)
    print('running')
    print(out)
