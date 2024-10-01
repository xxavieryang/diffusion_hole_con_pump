import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('t_e.csv')
data2 = pd.read_csv('e_w1.csv')
#data22 = pd.read_csv('s_w1.csv')
data3 = pd.read_csv('e_w10.csv')
data4 = pd.read_csv('e_w01.csv')

x1 = data1
y1 = data2
#y11 = data22
y10 = data3
y01 = data4

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 22})
#plt.ylim(9, 11.1)
plt.plot(x1, y1, label='D=1',linewidth=3)
#plt.plot(x1, y11, label='Point source',linewidth=3)
plt.plot(x1, y10, label='D=10',linewidth=3)
plt.plot(x1, y01, label='D=0.1',linewidth=3)
plt.title("Relative L2 error")
plt.xlabel('time') 
plt.legend()
plt.grid()
#plt.show()
plt.savefig("noconsumption.png")
