import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class KalmanFilter:
    def __init__(self, process_noise, measurement_noise, initial_state):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = initial_state
        self.covariance = np.eye(len(initial_state))
        
    def predict(self, A):
        # Predict state
        self.state = np.dot(A, self.state)
        # Predict covariance
        self.covariance = np.dot(np.dot(A, self.covariance), A.T) + self.process_noise
        
    def update(self, measurement, C):
        # Compute Kalman gain
        K = np.dot(np.dot(self.covariance, C.T), np.linalg.inv(np.dot(np.dot(C, self.covariance), C.T) + self.measurement_noise))
        # Update state estimate
        self.state = self.state + np.dot(K, (measurement - np.dot(C, self.state)))
        # Update covariance
        self.covariance = np.dot((np.eye(len(self.state)) - np.dot(K, C)), self.covariance)


# Function to parse data from a given file
def parse_data(filename):
    data = {
        'index': [],
        'sensor': [],
        'rescale': [],
        'deadzone': [],
        'release': [],
        'boundary': [],
        'debounced_state': [],
        'debounce_counter': []
    }
    
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                # Splitting directly after "Matthijs Muller:HE65:1: " and removing unnecessary parsing
                _, info = line.split("Matthijs Muller:HE65:1: ", 1)
                
                # Extract each item directly by splitting the remaining part
                parts = info.split(", ")
                index = int(parts[0].split(": ")[1])
                sensor = int(parts[1].split(": ")[1])
                rescale = int(parts[2].split(": ")[1])
                deadzone = int(parts[3].split(": ")[1])
                release = int(parts[4].split(": ")[1])
                boundary = int(parts[5].split(": ")[1])
                debounced_state = int(parts[6].split(": ")[1])
                debounce_counter = int(parts[7].split(": ")[1])
                
                # Append data to the lists
                data['index'].append(index)
                data['sensor'].append(sensor)
                data['rescale'].append(rescale)
                data['deadzone'].append(deadzone)
                data['release'].append(release)
                data['boundary'].append(boundary)
                data['debounced_state'].append(debounced_state)
                data['debounce_counter'].append(debounce_counter)
            except IndexError as e:
                print(f"Error parsing line {line_number}: {line.strip()} - {e}")
                continue
    
    return pd.DataFrame(data)

def plot_data(df_original, df_filtered):
    # Set up the figure size and resolution
    plt.figure(figsize=(14, 7), dpi=100)

    # Iterate over the DataFrame to change color based on debounced state and add transition lines
    previous_state = df_original['debounced_state'][0]
    start_index = 0

    for i in range(1, len(df_original)):
        current_state = df_original['debounced_state'][i]
        if current_state != previous_state:
            color = 'g' if previous_state == 0 else 'r'
            plt.plot(df_original.index[start_index:i], df_original['sensor'][start_index:i], color=color, linewidth=2)
            plt.plot(df_original.index[i-1:i+1], df_original['sensor'][i-1:i+1], color='y', linewidth=2, label='Debounce Transition' if 'Debounce Transition' not in plt.gca().get_legend_handles_labels()[1] else "")  # Yellow transition line
            start_index = i
            previous_state = current_state

    # Ensure the last segment is also plotted
    color = 'g' if df_original['debounced_state'].iloc[-1] == 0 else 'r'
    plt.plot(df_original.index[start_index:], df_original['sensor'][start_index:], color=color, linewidth=2)

    # Add a horizontal dotted line for the boundary value
    if not df_original['boundary'].empty:
        boundary_value = df_original['boundary'].iloc[0]  # Assuming the boundary value is the same for all samples
        plt.axhline(y=boundary_value, color='k', linestyle='--', linewidth=1, label='Deadzone')

    # Adding titles and labels
    plt.title('Sensor Readings Over Time')
    plt.xlabel('Sample Number')
    plt.ylabel('Sensor Value')

    # Show grid
    plt.grid(True)

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()


def main():
    # Parse data from file
    filename = "sensor_data.txt"
    df_original = parse_data(filename)
    df_filtered = df_original.copy()  # Make a copy to store filtered data
    
    # Initialize Kalman filter
    initial_state = np.array([0])  # Initial state (adjust dimensions as needed)
    process_noise = np.eye(1) * 0.01   # Process noise covariance matrix
    measurement_noise = np.eye(1) * 0.1  # Measurement noise covariance matrix
    kf = KalmanFilter(process_noise, measurement_noise, initial_state)
    
    # Apply Kalman filter to sensor data
    filtered_sensor_values = []  # Store filtered sensor values
    for i in range(len(df_original)):
        measurement = np.array([df_original['sensor'][i]])  # Sensor measurement
        C = np.eye(1)  # Measurement matrix (identity matrix for 1D measurement)
        kf.predict(A=np.eye(1))  # Assuming identity state transition matrix
        kf.update(measurement, C)
        filtered_sensor_values.append(kf.state[0])  # Store filtered value
    
    # Add filtered sensor values to the DataFrame
    df_filtered['filtered_sensor'] = filtered_sensor_values
    
    # Plot the original and filtered data
    plot_data(df_original, df_filtered)

if __name__ == "__main__":
    main()