import numpy as np
from datetime import datetime
from scipy.signal import butter,filtfilt
from sympy import symbols, solve

# Carbonator Geometry
# - Funnel https://drive.google.com/drive/folders/12Xoa6cEP1GAJa1TQKTZIqE5AyD2Jvw1F
FUNNEL_H = 1100 / 1000 #[m]
FUNNEL_R0 = 200 / 2 / 1000 #[m] 8 inch
FUNNEL_R1 = 900 / 2 / 1000 #[m] 36 inch
FUNNEL_V = FUNNEL_H * np.pi / 3 * (FUNNEL_R1 **2 + FUNNEL_R1 * FUNNEL_R0 + FUNNEL_R0 **2)
# - Cylinder
CROSSSEC_AREA = 0.6366 #[m^2]

# Lowpass Filter

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# string to datetime

INPUT_DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"
def str_to_datetime(s, from_=INPUT_DATETIME_FORMAT):
    """Convert a string to a datetime object"""
    return datetime.strptime(s, from_)

# Calculate delta in seconds between two points (requires %f format)

def delta_second(this_time,last_time):
    """Calculate the difference in seconds between two datetime strings"""
    this_micros = int(this_time[17:19]) + int(this_time[20:26])/1000000
    last_micros = int(last_time[17:19]) + int(last_time[20:26])/1000000

    delta = (this_micros-last_micros)
    if delta < 0:
        delta = delta + 60

    return delta

# calculate valve throughput for a given speed in Hz

def m_dot_valve(valve_speed):
    """"Calculate m_dot in kg/s"""
    m_dot = (13.8639057*valve_speed + 7.028552973)/3600
    return m_dot

# Process state, m_add into ms

def process_mass(original_df):
    """Process m_add data into the actual mass in the carbonator"""
    ms = 0
    m_available = 0
    delta_t = 0
    delta_m = 0

    length = original_df.shape[0]

    ms_list = [0]*length

    for index in range(1,length):
        
        state = original_df.loc[index].at["state"]
        m_add = original_df.loc[index].at["m_add"]

        delta_t = delta_second(original_df.loc[index].at["datetime"],original_df.loc[index-1].at["datetime"])

        # check if there's new mass available
        if m_add != 0:
            m_available = m_available + m_add

        # filling
        if state == 2:
            if m_available > 0:
                delta_m = m_dot_valve(50)*delta_t       # mass is in the hopper, add it
            else:
                delta_m = 0

        # emptying
        elif state == 3:
            if ms > 0:
                delta_m = -m_dot_valve(50)*delta_t      # mass is in the carbonator, remove it
            else:
                delta_m = 0

        # moving bed
        elif state == 4:
            if m_available > 0:     # if there's mass in the hopper, its a net positive delta for this case
                delta_m = (m_dot_valve(50)-m_dot_valve(13))*delta_t
            elif m_available == 0 and ms > 0:       # no mass in hopper, removing at slow rate
                delta_m = -m_dot_valve(13)*delta_t
            else:           # no mass anywhere, don't do anything
                delta_m = 0

        else:
            delta_m = 0

        # mass balancing
        if delta_m != 0:
            
            # case 1: trying to remove mass that doesn't exist
            if delta_m < 0 and abs(delta_m) > ms:
                delta_m = -m_available

            # case 2: trying to add mass that doesn't exist
            elif delta_m > 0 and delta_m > m_available:
                delta_m = m_available

            if delta_m > 0:     # carbonator can only gain mass if available subsequently decreases
                m_available = m_available - delta_m

            ms = ms + delta_m

        ms_list[index] = ms  # place current value into list

    return ms_list


## Smooth mass curve

def smooth_stepwise_inc_curve(from_time, to_time, timestamps, data, time_format):
    """Only works for monotonically increasing stepwise curve
    
    from_time : the timestamp from which the smoothing starts
    to_time : the timestamp where the smoothing stops
    timestamps, data: x and y coordinates of the curve to be smoothed
    time_format: the time format to use to parse from_time and to_time
    """
    from_time = datetime.strptime(from_time, time_format)
    to_time = datetime.strptime(to_time, time_format)
    # Get the index corresponding to the given value from timestamps
    start_index = timestamps[timestamps == from_time].index[0]
    end_index = timestamps[timestamps == to_time].index[0]

    # Trim timestamps and timeseries to [start_index, end_index]
    # trimmed_timestamps = timestamps[start_index:end_index]
    trimmed_data = list(data[start_index:end_index])
    trimmed_data_copy = list(data[start_index:end_index])

    # Find major points (the rightmost point of each horizontal line segment)
    major_indices = [len(trimmed_data)-1]
    major_points = []

    while len(trimmed_data)>0:
        major_points.append(np.max(trimmed_data))
        major_indices.append(np.argmax(trimmed_data)-1)
        trimmed_data = trimmed_data[:major_indices[-1]+1]

    major_indices[-1] = 0
    major_points.append(major_points[-1])
    major_indices.reverse()
    major_points.reverse()

    # Linear interpolation between major points
    list_buffer = [[major_points[0]]]
    for x0, x1, y0, y1 in zip(major_indices[:-1], major_indices[1:], major_points[:-1], major_points[1:]):
        line_seg = np.linspace(y0, y1, x1-x0+1)
        list_buffer.append(line_seg[1:])

    # Flatten list of lists
    smoothed_trimmed_data = list(np.concatenate(list_buffer).flat)

    # Visualize for debugging
    # import matplotlib.pyplot as plt
    # plt.plot(trimmed_data_copy)
    # plt.scatter(major_indices, major_points)
    # plt.plot(smoothed_trimmed_data)
    # plt.show()

    # Place the smoothed segment back to original data series
    smoothed_data = data.copy()
    smoothed_data[start_index:end_index] = smoothed_trimmed_data

    return smoothed_data

# Convert weight to volume for quicklime

def quicklime_weight_to_volume(w, bulkdensity=955):
    # bulk density in [kg/m^3]
    return w / bulkdensity

# Convert carbonator fill volume to fill height

def carbonator_fill_volume_to_height(fill_vol):
    """Given the volume of the carbonator filled, return the height of the carbonator filled"""
    if fill_vol > FUNNEL_V:
        # Above funnel-cylinder transition line
        additional_vol = fill_vol - FUNNEL_V
        additional_height = additional_vol / CROSSSEC_AREA
        return FUNNEL_H + additional_height
    else:
        # Below funnel-cylinder transition line
        subfunnel_h = symbols('h', real=True)
        subfunnel_r1 = FUNNEL_R0 + (FUNNEL_R1 - FUNNEL_R0) * (subfunnel_h / FUNNEL_H)
        subfunnel_v = subfunnel_h * np.pi / 3 * (subfunnel_r1 **2 + subfunnel_r1 * FUNNEL_R0 + FUNNEL_R0 **2)
        return float(max(solve(subfunnel_v - fill_vol, subfunnel_h)))
    
# Convert fill mass to carbonator fill height for quicklime

def solid_mass_to_fill_height(m):
    return carbonator_fill_volume_to_height(quicklime_weight_to_volume(m))

# Custom ".apply()" function: apply a function to each element of a series

def coarse_apply(series, func, n=300):
    """Similar to .apply() for Pandas Series but only apply func to n number of points in the 
    series and interpolate the points in between"""
    major_indices = np.linspace(0, len(series)-1, n).astype(int)
    results = np.zeros(len(series))
    prev_val = 0
    for i, j in enumerate(major_indices):
        results[j] = func(series[j])
        if i > 0:
            # linear interpolation
            interp = np.linspace(prev_val, results[j], j - major_indices[i-1] + 1)
            results[major_indices[i-1]:j] = interp[:-1]
        prev_val = results[j]
    return results

# Below is the Main Function

def add_calculated_columns(main_df, CONFIG):
    """Take in a Pandas DataFrame and a config dictionary. 
    
    Returns the DataFrame with added columns that contain calculated values
    e.g. pg_lowpass, po_lowpass, smooth_ms, fill_height, etc.
    
    """
    fs = float(CONFIG['data_rate'])

    cutoff = fs/80     # desired cutoff frequency of the filter, Hz
    order = 1 

    excluded_metrics = CONFIG["exclude_metrics"]
    METRICS_TO_APPLY_LOWPASS= ["pg", "po", "pv","pi"] + ["dpo", "dpf", "dpv"] + ["mdoti", "mdoto"]

    for metric_name in METRICS_TO_APPLY_LOWPASS:
        filtered_data = butter_lowpass_filter(main_df[metric_name], cutoff, fs, order)
        lowpass_name = metric_name + "_lowpass"
        main_df[lowpass_name] = filtered_data

    if "mass_smoothing_start_time" in CONFIG:
        main_df["smooth_ms"] = smooth_stepwise_inc_curve(
            CONFIG["mass_smoothing_start_time"], 
            CONFIG["mass_smoothing_end_time"],
            main_df["datetime"], 
            main_df["ms"],
            CONFIG["time_format"])
        # Apply function element-wise, use custom coarse_apply function to speed things up
        main_df["fill_height"] = coarse_apply(main_df["smooth_ms"], solid_mass_to_fill_height)
    
    elif "m_add" not in excluded_metrics:
        main_df['ms'] = process_mass(main_df)
        main_df["fill_height"] = coarse_apply(main_df["ms"], solid_mass_to_fill_height)
    
    
    # had to hard code the str_to_datetime
    
    main_df["datetime"] = main_df["datetime"].map(str_to_datetime)
    return main_df

def coarse_data(df, n=100):
    coarse_df = df[df.reset_index().index % n == 0]
    return coarse_df
        
        
        
###############

## Solve for D0 particle diameter
#
# def calculate_effective_particle_size(gas_mass_flow, temperature, CO2_concentration, pressure, fill_height, P0):
#     # P0: Pa, pressure drop induced when bed height = 0
#     # gas_mass_flow: kg/s
#     # temperature: °C
#     # CO2 concentration: fraction between 0 and 1
#     if fill_height < FUNNEL_H or pressure < P0:
#         print(fill_height, pressure)
#         return np.nan
# 
#     bed_void_space = 0.4
#     fluid_viscosity = 0.00004 # Pa.s
#     cross_sectional_area = 0.636 # m^2
#     delta_p_per_meter = (pressure - P0) / (fill_height - FUNNEL_H) # Pa
#     print(delta_p_per_meter)
# 
#     # Ergun's equation
#     dp_dL, m_dot, T, c_CO2, A, D_0 = symbols('dp/dL, \\dot{m}, T, c_{CO2}, A, D_0')
#     mu, epsilon = symbols('\\mu, \\epsilon')
#     rho = (c_CO2*(1*44/(0.0821*(T+273.15)))+(1-c_CO2)*(101.325*1000/(287.05*(T+273.15))))
#     u_0 = m_dot / (rho * A)
#     dp_dL = (150 * mu * (1-epsilon) ** 2 * u_0) / (epsilon **3 * D_0 ** 2) + (1.75 * (1 - epsilon) * rho * u_0 ** 2) / (epsilon ** 3 * D_0)
# 
#     return float(max(solve(dp_dL.subs(epsilon, bed_void_space)
#           .subs(mu, fluid_viscosity)
#           .subs(m_dot, gas_mass_flow)
#           .subs(T, temperature)
#           .subs(A, cross_sectional_area)
#           .subs(c_CO2, CO2_concentration) - delta_p_per_meter, D_0)))
# j = 10000
# print(main_df["mdoti_lowpass"][j], 
#                                   main_df["tc1"][j], 
#                                   main_df["gai"][j] / 100,
#                                   main_df["pg"][j] * 1000,
#                                   main_df["fill_height"][j])
# print(calculate_effective_particle_size(main_df["mdoti_lowpass"][j], 
#                                   main_df["tc1"][j], 
#                                   main_df["gai"][j] / 100,
#                                   main_df["pg"][j] * 1000,
#                                   main_df["fill_height"][j],
# 14 * 1000))