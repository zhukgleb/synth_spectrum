from data import extract_data                                                                 
import matplotlib.pyplot as plt
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl
import numpy as np


#s2 = extract_data("data/NES_model_110000.rgs", text=True)
#s3 = extract_data("data/NES_model_60000.rgs", text=True)
s4 = extract_data("data/NES_model_40000.rgs", text=True)
#s5 = extract_data("data/NES_model_15000.rgs", text=True)
#     s6 = extract_data("data/NES_model_30000.rgs", text=True)
a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

spectrum_arr = [s4]
spectrum_names = ["R=40000 inter"]
spectrum_names_direct = ["R=40000"]
total_velocity_data = []
total_delta = []
total_delta_inter = []

for i in range(len(spectrum_arr)):
    velocity = []
    z_velocity = []
    true_velocity = []
    delta = []
    delta_inter = []
    for j in range(10, 700, 10):
       print(f"speed is {j} meters")
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], 20/1000, edgeHandling="firstlast")
       mult = j
       cv, z = find_velocity(spectrum_arr[i], [a_template, f_template], [5000, 5100], mult)
       velocity.append(cv)
       z_velocity.append(z)
       true_velocity.append(j)
       delta_inter.append(20 - z)  # For delta graph
       delta.append(20 - cv)
       
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], -20/1000, edgeHandling="firstlast")
 

    velocity_data = [true_velocity, velocity, z_velocity]
    total_velocity_data.append(velocity_data)
    total_delta_inter.append(delta_inter)
    total_delta.append(delta)

# A very bad part. btw -- it's time to get it done
plt.style.use('seaborn-v0_8-talk')
plt.grid()
for i in range(len(spectrum_arr)):
    plt.plot(np.arange(10, 700, 10), total_delta_inter[i], label=spectrum_names[i])
    plt.plot(np.arange(10, 700, 10), total_delta[i], label=spectrum_names_direct[i])

plt.xlabel("Dots")
plt.ylabel("Delta")
# plt.yscale("log")
plt.legend()
plt.show()                                                                                    
