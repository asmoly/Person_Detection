import pandas as pd
import matplotlib.pyplot as plt
import requests

SUBPLOT_ROWS = 3
SUBPLOT_COLUMNS = 1

data_str = "https://eps-datalogger.herokuapp.com/api/data/sasha/person_counter_run3"

response = requests.get(data_str)
response_json = response.json()
df = pd.DataFrame.from_dict(response_json)

df["time_created"] = pd.to_datetime(df["time_created"])
df = df.sort_values(by="time_created")
df.index = df["time_created"]
df.index = df.index.tz_convert("US/Pacific")

fig = plt.figure(figsize=(12, 6))
fig.suptitle("Number of people")

# Line plot
line_plot = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLUMNS, 1)
line_plot.set_ylabel("People")

line_plot.plot(df["float1"])
line_plot.tick_params(labelrotation=0)

# Bar graph
school_hours_sum = 0
school_hours_count = 0

not_school_hours_sum = 0
not_school_hours_count = 0

for i in range (0, df.shape[0]):
    time_in_minutes = df.index[i].hour*60 + df.index[i].minute
    
    if time_in_minutes > 510 and time_in_minutes < 900:
        school_hours_sum += df["float1"][i]
        school_hours_count += 1
    else:
        not_school_hours_sum += df["float1"][i]
        not_school_hours_count += 1

x_axis = ["School Hours", "Non School Hours"]
y_axis = [school_hours_sum/school_hours_count, not_school_hours_sum/not_school_hours_count]

bar_plot = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLUMNS, 2)
bar_plot.set_ylabel("People (Average)")

bar_plot.bar(x_axis, y_axis)

# Bar Graph with Periods
periods = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range (0, df.shape[0]):
    time_in_minutes = df.index[i].hour*60 + df.index[i].minute
    
    if df.index[i].day == 23:
        if time_in_minutes > 830 and time_in_minutes < 900:
            periods[0] += df["float1"][i]
        elif time_in_minutes > 745 and time_in_minutes < 815:
            periods[1] += df["float1"][i]
        elif time_in_minutes > 595 and time_in_minutes < 665:
            periods[2] += df["float1"][i]
        elif time_in_minutes > 510 and time_in_minutes < 580:
            periods[3] += df["float1"][i]
    elif df.index[i].day == 24:
        if time_in_minutes > 830 and time_in_minutes < 900:
            periods[4] += df["float1"][i]
        elif time_in_minutes > 745 and time_in_minutes < 815:
            periods[5] += df["float1"][i]
        elif time_in_minutes > 595 and time_in_minutes < 665:
            periods[6] += df["float1"][i]
        elif time_in_minutes > 510 and time_in_minutes < 580:
            periods[7] += df["float1"][i]

x_axis = ["a", "b", "c", "d", "e", "f", "g", "h"]
y_axis = periods

periods_bar_plot = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLUMNS, 3)
periods_bar_plot.set_ylabel("People (Cumulative)")

periods_bar_plot.bar(x_axis, y_axis)

plt.show()