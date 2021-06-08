
f = open("/home/liuliang/data.txt")
line = f.readline()
line = line.strip('\n')
c = []
while line:
    # print(line)
    c.append(int(line))
    line = f.readline()
    line = line.strip('\n')
f.close()
c.sort()
len1 = int(len(c)/2)
print(c[len1])

