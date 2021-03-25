import rarfile
import os
rar = rarfile.RarFile("data_reduced.rar")
rar.extractall()
print("Extraction Done")