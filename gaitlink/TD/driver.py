from gaitlink.data import LabExampleDataset
from gaitlink.TurningDetection import TurningDetection
import pandas as pd
from pandas import unique

# Load example data
data = LabExampleDataset()

# Define the unique values for each level of data
cohort_values = data.subset_index['cohort']
participant_values = data.subset_index['participant_id']
test_values = data.subset_index['test']
trial_values = data.subset_index['trial']

# Initialise empty list
results_list = []

# Set to store processed combinations
processed_combinations = set()

# Main method
turning_detector = TurningDetection()

# Loop across different data levels
for cohort in cohort_values:
    for participant in participant_values:
        for test in test_values:
            for trial in trial_values:
                # Create a unique combination key
                combination_key = (cohort, participant, test, trial)

                # Check if this combination has already been processed
                if combination_key in processed_combinations:
                    continue

                # Add the combination to the set to mark it as processed
                processed_combinations.add(combination_key)

                # Construct the subset index to access data for the current combination
                subset_data = data[
                    (data.subset_index['cohort'] == cohort) &
                    (data.subset_index['participant_id'] == participant) &
                    (data.subset_index['test'] == test) &
                    (data.subset_index['trial'] == trial)
                    ]

                # Check if there's only a single combination left in the subset
                if len(subset_data) == 1:
                    data_ = subset_data.data['LowerBack'].gyr_z # Provide the gyro data here
                    fs_hz = 100  # Provide the sampling frequency here

                    # Detect turns
                    turning_detector.detect_turning_points(data_, fs_hz)

                    # Extract turning information
                    gyr_z_lp = turning_detector.gyr_z_lp # filtered gyroscope z
                    turning_list = turning_detector.turning_list # list of turns

                    # Post process
                    turning_detector.post_process(turning_list, fs_hz, gyr_z_lp)

                    # Store the values or arrays in a tuple or a dictionary
                    result = {
                        'cohort': cohort,
                        'participant': participant,
                        'test': test,
                        'trial': trial,
                        'turning_list': turning_detector.turning_list,
                        'Turn_Angles_Degrees': turning_detector.all_angles_degrees,
                        'Turn_Duration_seconds': turning_detector.duration_list_seconds,
                        'TurnStart_seconds': turning_detector.Turn_Start_seconds,
                        'TurnEnd_seconds': turning_detector.Turn_End_seconds
                    }

                    # Append the result to the list
                    results_list.append(result)

# Convert the list of attributes into a DataFrame
attributes_df = pd.DataFrame(results_list)

columns_to_explode = ['Turn_Angles_Degrees', 'Turn_Duration_seconds', 'TurnStart_seconds', 'TurnEnd_seconds']

# Expand Turn_Angles_Degrees into separate rows
attributes_df = attributes_df.explode(columns_to_explode).reset_index(drop=True)




turns = turning_detector.Turn_End_seconds



