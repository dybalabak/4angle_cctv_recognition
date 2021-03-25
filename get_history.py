
import os
import matplotlib.pyplot as plt
import pandas as pd
#os.listdir("save_weight/")
def get_history():
    data = []
    for file in os.listdir("save_weight/"):
        if ".csv" not in file:
            file = file[21:]
            file = file[:-5]
            file = file.split("-")
            data.append(file)

        #print(file)
    #name = "history1"
    name = "history"
    data = pd.DataFrame(data, columns = ["epochs", "training_acc", "testing_acc"])
    data.to_csv("save_weight/"+name+".csv", index= False)
    data = pd.read_csv("save_weight/"+name+".csv")
    data = data.sort_values("epochs", ascending = True)
    data.to_csv("save_weight/"+name+".csv", index= False)
    data.info()
    return data
data = get_history()
data.index = data["epochs"]
data[["training_acc", "testing_acc"]].plot()
plt.title("Training on InceptionResnetV2 architecture")
plt.savefig("images/Training_and_Validation_graph.png")
plt.show()
def delete_weights():
    for file in os.listdir("save_weight/"):
        if "history" in file or "400" in file or "550" in file or "600" in file or "610" in file or "620" in file:           
            pass
            
        else:
            os.remove("save_weight/"+ file)
#delete_weights()