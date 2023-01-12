import pandas as pd

d = {'col1': [1, 2], 'col2': [3, 4], 'col3':[5,6]}
df = pd.DataFrame(data=d)
# print(df)
# print(df['col1'].tolist())

# my_enum = enumerate(zip(df["col1"].tolist(),df["col2"].tolist()))

# for index, (col1, col2) in my_enum:
#     print(index,col1, col2)


def m_dot_valve(valve_speed):
    """"Calculate m_dot in kg/s"""
    m_dot = (13.8639057*valve_speed + 7.028552973)/3600
    return m_dot


print(m_dot_valve(50)*3600)