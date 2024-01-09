import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
import csv
import math
import os
import argparse
############################################################################
# Script what plots ground truth coordinates and calculated coordinates 
############################################################################

def mse(actual, predicted):
    return (np.square(np.subtract(np.array(actual), np.array(predicted)))).mean()

def dif(actual, predicted):
    return np.subtract(np.array(actual), np.array(predicted))


filename_method1 = '../results/calculated_coordinates_method1.csv'
filename_method2 = '../results/calculated_coordinates_method2.csv'

data_method1 = pd.read_csv(filename_method1)
data_method2 = pd.read_csv(filename_method2)
imagesNum_total = data_method1.shape[0]
data_method1_location = data_method1.loc[data_method1['Calculated_Latitude'] != -1]
data_method2_location = data_method2.loc[data_method2['Calculated_Latitude'] != -1]
imagesNum_method1_location = data_method1_location.shape[0]
imagesNum_method2_location = data_method2_location.shape[0]

dataset_list_method1 = []
for i in range(0,data_method1_location.shape[0]):
    dataset_list_method1.append("Dataset 1")
data_method1_location['Dataset'] = dataset_list_method1

dataset_list_method2 = []
for i in range(0,data_method2_location.shape[0]):
    dataset_list_method2.append("Dataset 1")
data_method2_location['Dataset'] = dataset_list_method2

pd_zeros_method1 = pd.DataFrame(np.zeros(data_method1_location.shape[0]))
pd_zeros_method2 = pd.DataFrame(np.zeros(data_method2_location.shape[0]))
 
print("Total images ", imagesNum_total) 
print("imagesNum_method1_location", imagesNum_method1_location) 
print("imagesNum_method2_location", imagesNum_method2_location)
#print("True_images %", located_images *100 / True_images)
#print("Located images %", located_images *100 / total_images)

# Using DataFrame.mean() method to get column average
MAE_meter_method1 = mean_absolute_error(pd_zeros_method1, data_method1_location['Meters_Error'], )
MAE_meter_method2 = mean_absolute_error(pd_zeros_method2, data_method2_location['Meters_Error'], )
RMSE_meter_method1 = mean_squared_error(pd_zeros_method1, data_method1_location['Meters_Error'], squared=False)                                            
RMSE_meter_method2 = mean_squared_error(pd_zeros_method2, data_method2_location['Meters_Error'], squared=False)     

print("MAE_meter_method1:", MAE_meter_method1) 
print("MAE_meter_method2:", MAE_meter_method2)   
print("RMSE_meter_method1:", RMSE_meter_method1)    
print("RMSE_meter_method2:", RMSE_meter_method2)

MAE_Latitude_method1 = mean_absolute_error(data_method1_location['Latitude'], data_method1_location['Calculated_Latitude'], )
MAE_Longitude_method1 = mean_absolute_error(data_method1_location['Longitude'], data_method1_location['Calculated_Longitude'], )
RMSE_Latitude_method1 = mean_squared_error(data_method1_location['Latitude'], data_method1_location['Calculated_Latitude'], squared=False)
RMSE_Longitude_method1 = mean_squared_error(data_method1_location['Longitude'], data_method1_location['Calculated_Longitude'], squared=False)

MAE_Latitude_method2 = mean_absolute_error(data_method2_location['Latitude'], data_method2_location['Calculated_Latitude'], )
MAE_Longitude_method2 = mean_absolute_error(data_method2_location['Longitude'], data_method2_location['Calculated_Longitude'], )
RMSE_Latitude_method2 = mean_squared_error(data_method2_location['Latitude'], data_method2_location['Calculated_Latitude'], squared=False)
RMSE_Longitude_method2 = mean_squared_error(data_method2_location['Longitude'], data_method2_location['Calculated_Longitude'], squared=False)

print("MAE_Latitude_method1:", MAE_Latitude_method1)
print("MAE_Longitude_method1:", MAE_Longitude_method1)
print("RMSE_Latitude_method1:", RMSE_Latitude_method1)
print("RMSE_Longitude_method1:", RMSE_Longitude_method1)

print("MAE_Latitude_method2:", MAE_Latitude_method2)
print("MAE_Longitude_method2:", MAE_Longitude_method2)
print("RMSE_Latitude_method2:", RMSE_Latitude_method2)
print("RMSE_Longitude_method2:", RMSE_Longitude_method2)


#print(data2)

fig=plt.figure(figsize=(9,6))
ax=plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.patch.set_facecolor('white')
ax.patch.set_alpha(0.4)
sns.set(context = None,style='ticks',font_scale=1.2)
plt.grid(color='lightgray')
#Experiment 1
sns.lineplot(data = data_method1_location, x = data_method1_location.index , y = "Meters_Error",label = 'Projection_center_without_IME', marker="o", color='r',linewidth = 1.5)
sns.lineplot(data = data_method2_location, x = data_method2_location.index , y = "Meters_Error",label ='Center_projection_without_IME', marker="d", color='b',linestyle='--',linewidth = 1.5)
#Experiment 2
#sns.lineplot(data = data_method1_location, x = data_method1_location.index , y = "Meters_Error",label = 'Center_projection_without_IME',  marker="o", color='r' ,linewidth = 1.5 )
#sns.lineplot(data = data_method2_location, x = data_method2_location.index , y = "Meters_Error",label ='Center_projection_with_IME(no_area)', marker="d", color='b',linestyle='--',linewidth = 1.5)
#Experiment 3
#sns.lineplot(data = data_method1_location, x = data_method1_location.index , y = "Meters_Error",label = 'Center_projection_with_IME(no_area)',marker="o", color='r',linewidth = 2.5)
#sns.lineplot(data = data_method2_location, x = data_method2_location.index , y = "Meters_Error",label ='Center_projection_with_IME', marker="d", color='b',linestyle='--',linewidth =  1.5)
plt.tick_params(labelsize=12)         #刻度值字体大小设置（x轴和y轴同时设置）
ax.set_xlabel("Image Index",fontsize=14)
ax.set_ylabel("Localization Error(m)",fontsize=14)
plt.savefig("mean_absolute_error.png")
plt.savefig("mean_absolute_error.pdf")


fig=plt.figure(figsize=(9,6))
ax=plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.patch.set_facecolor('white')
ax.patch.set_alpha(0.4)
sns.set(context = None,style='ticks',font_scale=1.2)     
plt.grid(color='lightgray')
#Experiment 1
sns.lineplot(data = data_method1_location, x = "Longitude_Error" , y = "Latitude_Error" , label = 'Projection_center_without_IME',marker="o", color='r',linewidth = 1.5)
sns.lineplot(data = data_method2_location, x = "Longitude_Error" , y = "Latitude_Error" , label ='Center_projection_without_IME', marker="d", color='b',linestyle='--',linewidth = 1.5)
#Experiment 2
#sns.lineplot(data = data_method1_location, x = "Longitude_Error" , y = "Latitude_Error" , label = 'Center_projection_without_IME',  marker="o", color='r',linewidth = 1.5)
#sns.lineplot(data = data_method2_location, x = "Longitude_Error" , y = "Latitude_Error" , label ='Center_projection_with_IME(no_area)', marker="d", color='b',linestyle='--',linewidth = 1.5)
#Experiment 3
#sns.lineplot(data = data_method1_location, x = "Longitude_Error" , y = "Latitude_Error" , label = 'Center_projection_with_IME(no_area)',marker="o", color='r',linewidth = 2.5)
#sns.lineplot(data = data_method2_location, x = "Longitude_Error" , y = "Latitude_Error" , label ='Center_projection_with_IME', marker="d", color='b',linestyle='--',linewidth =  1.5)
plt.tick_params(labelsize=12)      
ax.set_xlabel("Longitude Error(°)",fontsize=14)
ax.set_ylabel("Latitude Error(°)",fontsize=14)
plt.savefig("Longitude_Latitude_Error.png")
plt.savefig("Longitude_Latitude_Error.pdf")

plt.show()
