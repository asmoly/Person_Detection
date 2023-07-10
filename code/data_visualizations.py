import pandas as pd
import matplotlib.pyplot as plt
import requests

# Creates variables for the amount of rows and columns in the figure
SUBPLOT_ROWS = 3
SUBPLOT_COLUMNS = 1

# This is the link to the data
data_str = "https://eps-datalogger.herokuapp.com/api/data/sasha/person_counter_run3"

# This gets the data from the link and turns it into a data frame
response = requests.get(data_str)
response_json = response.json()
df = pd.DataFrame.from_dict(response_json)

# This converts the time column to date time object
df["time_created"] = pd.to_datetime(df["time_created"])
# This sorts the column by time
df = df.sort_values(by="time_created")
# Sets the index to the time
df.index = df["time_created"]
# Covnerts it to pasific time zone
df.index = df.index.tz_convert("US/Pacific")

# Creates a figure with size 12, 6
fig = plt.figure(figsize=(12, 6))
# Adds a title to the figure
fig.suptitle("Number of people")

# Line plot:
# Creates a subplot for the line plot
line_plot = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLUMNS, 1)
# Creates a y label for the subplot
line_plot.set_ylabel("People")

# This plots the float1 column which has the data for the number of people
line_plot.plot(df["float1"])

# Bar graph

# This creates variables for the sum of school hours and count of recorded data school hours
# these will be used to calculate the average by dividing the two variables
school_hours_sum = 0
school_hours_count = 0
# This is the same as the previous variables but it is for non school hours
not_school_hours_sum = 0
not_school_hours_count = 0

# This loops through the entire data frame
for i in range (0, df.shape[0]):
    # This converts the time into minutes by multiplying the hours
    # by 60 and adding the minutes
    time_in_minutes = df.index[i].hour*60 + df.index[i].minute
    
    # This checks if the time is within school hours
    if time_in_minutes > 510 and time_in_minutes < 900:
        # If it is then it adds the number of people at that time
        school_hours_sum += df["float1"][i]
        # And increments the count variable
        school_hours_count += 1
    else:
        # Otherwise it does the same but for the non school variables
        not_school_hours_sum += df["float1"][i]
        not_school_hours_count += 1

# This sets the x axis for the bar graph to just two strings
x_axis = ["School Hours", "Non School Hours"]
# This sets the y axis by doing the sum divided by the count to get the average
y_axis = [school_hours_sum/school_hours_count, not_school_hours_sum/not_school_hours_count]

# The creates a subplot for the bar graph
bar_plot = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLUMNS, 2)
# This sets the y label for the subplot
bar_plot.set_ylabel("People (Average)")

# This graphs the bar plot with the x axis and y axis
bar_plot.bar(x_axis, y_axis)

# Bar Graph with Periods
# This is a list that will be used to count people per period
# it goes in order a, b, c, d, e, f, g, h
periods = [0, 0, 0, 0, 0, 0, 0, 0]

# THis loops through the entire data frame
for i in range (0, df.shape[0]):
    # This converts the time to minutes
    time_in_minutes = df.index[i].hour*60 + df.index[i].minute
    
    # If the day is the 23rd then it checks for a, b, c, and d periods
    if df.index[i].day == 23:
        # This just checks what time frame the data is in and increments that period
        if time_in_minutes > 830 and time_in_minutes < 900:
            periods[0] += df["float1"][i]
        elif time_in_minutes > 745 and time_in_minutes < 815:
            periods[1] += df["float1"][i]
        elif time_in_minutes > 595 and time_in_minutes < 665:
            periods[2] += df["float1"][i]
        elif time_in_minutes > 510 and time_in_minutes < 580:
            periods[3] += df["float1"][i]
    elif df.index[i].day == 24:
        # This does the same on e, f, g, and h and on the 24th
        if time_in_minutes > 830 and time_in_minutes < 900:
            periods[4] += df["float1"][i]
        elif time_in_minutes > 745 and time_in_minutes < 815:
            periods[5] += df["float1"][i]
        elif time_in_minutes > 595 and time_in_minutes < 665:
            periods[6] += df["float1"][i]
        elif time_in_minutes > 510 and time_in_minutes < 580:
            periods[7] += df["float1"][i]

# This creates the x axis and sets the y axis to the periods list
x_axis = ["a", "b", "c", "d", "e", "f", "g", "h"]
y_axis = periods

# This creates a subplot for the bar graph
periods_bar_plot = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLUMNS, 3)
# This sets the y label
periods_bar_plot.set_ylabel("People (Cumulative)")

# This plots the bar graph
periods_bar_plot.bar(x_axis, y_axis)

# This shows all the graphs
plt.show()