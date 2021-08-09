'''
Date: 2021-06-25 15:22:31
LastEditors: Liuliang
LastEditTime: 2021-06-25 16:49:43
Description: read_file
'''
import matplotlib.pyplot as plt

with open('file.txt') as f:
    data = f.readline()
    # print(data)

print(data)
m = data.split(',')
m_2 = []
for i in m:
    i_1 = i.strip(' [').strip(']')
    m_2.append(float(i_1))

# print(m_2)
flops,acc = m_2[0:len(m_2)//2], m_2[len(m_2)//2:]
flops = [i/1e6 for i in flops ]

plt.plot(flops,acc)
plt.title("flops vs acc")
plt.xlabel("Flops(M)")
plt.ylabel("Acc(%)")
plt.savefig("./1.jpg")
plt.show()




