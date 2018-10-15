import pandas as pd
import matplotlib.pyplot as plt

path_licor = "/Licor/20180821.TXT"
path_kduino_mod1 = r"C:\Users\electrolabtop2\Google Drive\MONOCLE\Scotland" +
"field campaing\Data\Stirling\Profiler\KdUINO mod1\data_modified.txt"

raw = pd.read_csv(path_kduino_mod1, sep=" ", header=None)

red = []
green = []
blue = []
clear = []
# print(raw.head())
for i in range(len(raw.index)):

    for j in range(len(raw.columns)):
                if j < 2:
                    continue
                if j % 2 == 0:
                    if j % 4 == 0:
                        continue
                    red.append(raw[j].iloc[i])
                    green.append(raw[j+1].iloc[i])
                    blue.append(raw[j+2].iloc[i])
                    clear.append(raw[j+3].iloc[i])

print(clear)
plt.plot(clear)
plt.show()
