import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"data_fa\processed\79_A_1.csv")

print(df)

df = df[df["id"]==128]

a = plt.plot(df["x"],df["x_instant_speed"])
plt.savefig("test.png")