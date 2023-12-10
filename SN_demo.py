# Signal Generation
# matplotlib inline

from data import extract_data
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl
import numpy as np
import matplotlib.pyplot as plt
from data import test_data


#x, y = test_data()
#snr = 1
#noise = np.random.normal(loc=1, scale=1/snr, size=y.shape)
#noisy_signal = y + noise
#plt.plot(x, noisy_signal)
#plt.show()


s2 = extract_data("data/NES_model_110000.rgs", text=True)
# s3 = extract_data("data/NES_model_60000.rgs", text=True)
# s4 = extract_data("data/NES_model_40000.rgs", text=True)
s5 = extract_data("data/NES_model_15000.rgs", text=True)
a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

spectrum_arr = [s2, s5] 
spectrum_names = ["R=110000 inter", "R=15000 inter"]
spectrum_names_direct = ["R=110000","R=15000"]


total_velocity_data = []
total_delta = []
total_delta_inter = []

for i in range(len(spectrum_arr)):
    velocity = []
    z_velocity = []
    SN = []
    delta = []
    delta_inter = []
    # Now, make a variance between arrays -- add some noise
    # from SN 1 to 100
    for j in range(1, 100):
       print(f"speed is {j} meters")
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], 20/1000, edgeHandling="firstlast")
       
       noise_spectrum = np.copy(spectrum_arr[i][1])
       noise = np.random.normal(loc=1, scale=1/j, size=len(spectrum_arr[i][1]))
       noise_spectrum = noise_spectrum + noise

       cv, z = find_velocity([spectrum_arr[i][0], noise_spectrum], [a_template, f_template], [5000, 5100], 40)
       velocity.append(cv)
       z_velocity.append(z)
       SN.append(j)
       delta_inter.append(20 - z)  # For delta graph
       delta.append(20 - cv) 
    
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], -20/1000, edgeHandling="firstlast")
 

    velocity_data = [velocity, z_velocity]
    total_velocity_data.append(velocity_data)
    total_delta_inter.append(delta_inter)
    total_delta.append(delta)
    

# A very bad part. btw -- it's time to get it done
plt.style.use('seaborn-v0_8-talk')
plt.grid()
for i in range(len(spectrum_arr)):
    plt.plot(SN, total_delta[i], label=spectrum_names_direct[i])

plt.xlabel("S/N")
plt.ylabel("Delta")
# plt.yscale("log")
plt.legend()
plt.show()
