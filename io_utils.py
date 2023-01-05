import os
import csv

def get_test_dates(root_path=""):
    data_files = os.listdir(os.path.join(root_path,"data_analysis"))
    dates = []
    print("Available Dates:\n")
    for data_file in data_files:
        if data_file.split("_")[0] in ["EventLog", "SensorData"]:
            date = (data_file.split("_",1)[1]).split(".")[0]
            date = date[0:10]    # remove _processed if present
            if date not in dates:
                dates.append(date)
    dates.sort()
    for date in dates:
        print(date)
    return dates
    
def check_test_date(date, dates):
    if date not in dates:
        raise SystemExit("Invalid test date chosen, select from available dates shown")
    else:
        data_path = os.path.join("data_analysis", ("SensorData_"+date+".csv"))
        processed_data_path = os.path.join("data_analysis", ("SensorData_"+date+"_Processed.csv"))
        event_path = os.path.join("data_analysis", ("EventLog_"+date+".csv"))
        config_path = os.path.join("data_analysis", ("Config_"+date+".csv"))

    return (data_path,processed_data_path,event_path,config_path)

def config_to_dict(date):
    csv_path = os.path.join("data_analysis",("Config_"+date+".csv"))
    my_dict = {}

    with open(csv_path, 'r',encoding='utf-8-sig') as data:
        csv_reader = csv.reader(data)
        for row in csv_reader:
            my_dict[row[0]] = row[1]
        my_dict['exclude_metrics'] = my_dict['exclude_metrics'].split(",")      # list of metrics to exclude
    
    return my_dict
    
if __name__ == "__main__":
    # Test script
    test_dates = get_test_dates()
    test_date = "2022-12-01" #<<< Enter the Test Date Here
    (data_log_path,event_log_path,config_file_path) = check_test_date(test_date,test_dates)
    config = config_to_dict(test_date)