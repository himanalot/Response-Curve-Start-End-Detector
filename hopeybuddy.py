import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_response_start_end(file_path):
    # Load the data
    data = pd.read_csv(file_path, delimiter='\t')
    data['Datetime'] = pd.to_datetime(data['Time'])
    
    # Calculate the first and second derivatives of the 'rH' curve
    data['rH_derivative'] = np.gradient(data['rH'])
    data['rH_2nd_derivative'] = np.gradient(data['rH_derivative'])
    
    # Calculate the threshold as 3 times the standard deviation of the second derivative
    threshold_2nd = 3 * data['rH_2nd_derivative'].std()
    
    # Identify the points where the second derivative exceeds the threshold
    response_starts_2nd = data[data['rH_2nd_derivative'] > threshold_2nd]
    
    # Extract the first start point and move it to the local minimum before it
    first_start_index = response_starts_2nd.index[0]
    window_size = 5  # Define the window size for the "local" region
    local_region = data.loc[max(0, first_start_index - window_size) : first_start_index]
    first_start_index = local_region['rH'].idxmin()
    first_start = data.loc[[first_start_index]]
    
    # Calculate the difference between each 'rH' value and the next one
    data['rH_diff'] = data['rH'].diff()
    
    # Identify the points where this difference is below a certain negative threshold
    response_ends = data[data['rH_diff'] < -0.1]

    # Find the index of the absolute minimum of the first derivative
    min_derivative_index = data['rH_derivative'].abs().idxmin()
    
    # Calculate the difference between each 'rH' value and the next one
    data['rH_diff'] = data['rH'].diff()

    # Find the index of the drop-off point
    dropoff_index = data[data['rH_diff'] < -0.1].index[0]

    # Define the region after the drop-off point
    region = data.loc[dropoff_index:]

    # Find the local maxima in 'rH' within the region
    local_maxima = region[(region['rH'].shift(1) < region['rH']) & (region['rH'].shift(-1) < region['rH'])]

    # Find the last significant local maximum
    end_point_index = local_maxima[local_maxima['rH'] > 0.9 * local_maxima['rH'].max()].index[-1]
    response_end = data.loc[[end_point_index]]


    
    # Plot 'rH' and the identified start and end points of the response over time
    fig, ax1 = plt.subplots(figsize=(10,6))

    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('rH', color=color)
    ax1.plot(data['Datetime'], data['rH'], color=color)
    ax1.plot(first_start['Datetime'], first_start['rH'], 'bo', markersize=10, zorder=3)
    ax1.plot(response_end['Datetime'], response_end['rH'], 'rx', markersize=10, zorder=3)
    ax1.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
    
    return first_start, response_end

# Use the function on a given file
start, end = get_response_start_end(input("filepath"))

print("Start point:")
print(start)
print("End point:")
print(end)
