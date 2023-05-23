import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("Visualizations/data.json")

df["time_created"] = pd.to_datetime(df["time_created"])
df = df.sort_values(by="time_created")
df["time_created"] = df["time_created"].dt.tz_convert("US/Pacific")
df.index = df["time_created"]

plt.plot(df["float1"])

plt.title("Number of people")
plt.xticks(rotation=90)

plt.show()