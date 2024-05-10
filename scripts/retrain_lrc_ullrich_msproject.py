from mobgap.data import MsProjectDataset

# new_dataset = MsProjectDataset(base_path="./").get_subset(test="Test11")
new_dataset = MsProjectDataset(
    base_path="C:/Users/syrin/Documents/Studium/PhD/MS_Project/MsProject_data/Data", reference_system="SU_LowerShanks"
)

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
single_datapoint.data_ss  # [["gyr_x", "gyr_y", "gyr_z"]].plot(subplots=True)
# plt.show()
