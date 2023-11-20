import pandas as pd
from gatilink.data_transform._savgol_filter import SavgolFilter

# Create sample data
sample_data = pd.DataFrame({
    "sensor1": [1.0, 2.0, 3.0, 4.0],
    "sensor2": [0.5, 1.0, 1.5, 2.0],
})

# Create an instance of SavgolFilter
savgol_filter = SavgolFilter(window_length=5, polyorder=2)

# Apply the filter to the sample data
smoothed_data = savgol_filter.transform(sample_data)

# Access the transformed data
transformed_data = savgol_filter.transformed_data_

# Print or use the transformed data as needed
print(transformed_data)
