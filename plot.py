import matplotlib.pyplot as  plt 
import pandas as pd
import numpy as np
radius = pd.read_csv("Pas d'echange.csv")
radius2 = pd.read_csv("Echange d'episode.csv")
#radius3 = pd.read_csv("50.csv")
#radius4 = pd.read_csv("100.csv")




fig = plt.figure(figsize=(10,10))
colors = ['blue','green','red','cyan','magenta','yellow','black', 'grey', 'brown', 'olive','orange']


lis = radius["y.1"].to_list()
x = []
y = []
for i in range(50):
    x.append(int(i*(len(lis)/50)))
    r = np.mean(lis[   max(0,int(i*(len(lis)/50))-25):min(len(lis),int(i*(len(lis)/50))+25)    ])
    y.append(r)



lis2 = radius2["y.1"].to_list()
x2 = []
y2 = []
for i in range(50):
    x2.append(int(i*(len(lis2)/50)))
    r = np.mean(lis2[   max(0,int(i*(len(lis2)/50))-25):min(len(lis2),int(i*(len(lis2)/50))+25)    ])
    y2.append(r)

'''
lis3 = radius3["y.1"].to_list()
x3 = []
y3 = []
for i in range(50):
    x3.append(int(i*(len(lis3)/50)))
    r = np.mean(lis3[   max(0,int(i*(len(lis3)/50))-25):min(len(lis3),int(i*(len(lis3)/50))+25)    ])
    y3.append(r)
    
lis4 = radius4["y.1"].to_list()
x4 = []
y4 = []
for i in range(50):
    x4.append(int(i*(len(lis4)/50)))
    r = np.mean(lis4[   max(0,int(i*(len(lis4)/50))-25):min(len(lis4),int(i*(len(lis4)/50))+25)    ])
    y4.append(r)

'''
plt.plot(x, y, color=colors[0], marker="o", label="Normal agents")
plt.legend()


plt.plot(x2, y2, color=colors[1], marker="o", label="Averaging frequency 20")
plt.legend()

'''
plt.plot(x3, y3, color=colors[2], marker="o", label="Averaging frequency 50")
plt.legend()

plt.plot(x4, y4, color=colors[3], marker="o", label="Averaging frequency 100")
plt.legend()

plt.xlabel('Episode', fontsize=18)
plt.ylabel("Nombre d'etapes par Episode", fontsize=16)
'''
plt.show()