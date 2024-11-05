from mobgap.data import MsProjectDataset
import matplotlib.pyplot as plt

#new_dataset = MsProjectDataset(base_path="./").get_subset(test="Test11")
new_dataset = MsProjectDataset(base_path="C:/Users/syrin/Documents/Studium/PhD/MS_Project/MsProject_data/Data", reference_system="SU_LowerShanks")

print(new_dataset)

# %%
# Note: this works
# for d in new_dataset:
#     print(d)


# %%
# Note: this works
single_datapoint = new_dataset[0]
# print(single_datapoint)

# %%
# Note: this works
single_datapoint.data_ss#[["gyr_x", "gyr_y", "gyr_z"]].plot(subplots=True)
# plt.show()
# %%
print("trying single_datapoint.reference_parameters_.ic_list")
print(single_datapoint.participant_metadata)
print(single_datapoint.reference_parameters_.ic_list)



# %%

from mobgap.lrc.pipeline import LrcEmulationPipeline
from mobgap.lrc import LrcUllrich

pipe = LrcEmulationPipeline(LrcUllrich(**LrcUllrich.PredefinedParameters.untrained_svc)).run(single_datapoint)

# %%
from tpcp.optimize import GridSearch, GridSearchCV
from tpcp.validate import cross_validate
from sklearn.model_selection import ParameterGrid

para_grid = ParameterGrid({"algo__smoothing_filter__order": [1, 2, 3]})

grid_search = GridSearchCV(pipe, para_grid, return_optimized="accuracy")
grid_search.optimize(new_dataset)

# %%
import pandas as pd

results = pd.DataFrame(grid_search.gs_results_)
print(results["accuracy"])
# %%

best_performer = grid_search.run(single_datapoint)
