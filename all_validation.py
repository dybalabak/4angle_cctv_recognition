import os
import pandas as pd
validation_accuracies = []
weight_name = "weight-data-20201124-110-0.58-0.75.hdf5"# THIS is initial weight
weight_names = []
weight_epochs = []
for current_weight in os.listdir("save_weight"):
    if ".hdf5" in current_weight:
        print(current_weight)
        weight_name = current_weight
        
        exec(open("evaluate_model.py").read())
        weight_epochs.append(weight_name.split("-")[3])
        validation_accuracies.append(accuracy)
        weight_names.append(weight_name)
validation_df = pd.DataFrame(columns = ["Epochs", "Weight", "Final_validation_accurary"])
validation_df["Weight"] =weight_names
validation_df["Epochs"] = weight_epochs
validation_df["Final_validation_accurary"] = validation_accuracies
validation_df.to_csv("validation_accuracies.csv", index =False)
print(validation_df)