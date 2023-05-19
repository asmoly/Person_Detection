import requests

def data_logger_string(username, device_id, people_count):
    # It then creates the string where the first part is the link to the server
    # and then it adds all the parameter values in a specific formatt
    datalogger_url_str = "https://eps-datalogger.herokuapp.com/api/data/" \
    + username + "/add?device_id=" + device_id + "&int1=" + str(people_count)

    return datalogger_url_str


data = data_logger_string("sasha", "network_test2", 11)

test = requests.post(data)