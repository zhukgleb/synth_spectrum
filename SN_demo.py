from data import extract_data
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl
import numpy as np
import matplotlib.pyplot as plt
from data import test_data


# s2 = extract_data("data/NES_model_110000.rgs", text=True)
s3 = extract_data("data/NES_model_60000.rgs", text=True)
# s4 = extract_data("data/NES_model_40000.rgs", text=True)
s5 = extract_data("data/NES_model_15000.rgs", text=True)
a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

spectrum_arr = [s3, s5] 
spectrum_names = ["R=60000 inter", "R=15000 inter"]
spectrum_names_direct = ["R=60000","R=15000"]


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
    for j in range(100, 101):
       print(f"speed is {j} meters")
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], 20/1000, edgeHandling="firstlast")
       
       noise_spectrum = np.copy(spectrum_arr[i][1])
       noise = np.random.normal(loc=1, scale=1/j, size=len(spectrum_arr[i][1]))
       noise_spectrum = noise_spectrum + noise
#       import matplotlib.pyplot as plt
#       plt.plot(spectrum_arr[i][0], noise_spectrum)
#       plt.show()

       cv, z = find_velocity([spectrum_arr[i][0], noise_spectrum], [a_template, f_template], [5000, 5100], 50)
       velocity.append(cv)
       z_velocity.append(z)
       SN.append(j)
       delta_inter.append(20 - z)  # For delta graph
       delta.append(20 - cv) 
    
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], -20/1000, edgeHandling="firstlast")
       del noise_spectrum
 

    velocity_data = [velocity, z_velocity]
    total_velocity_data.append(velocity_data)
    total_delta_inter.append(delta_inter)
    total_delta.append(delta)
    

# A very bad part. btw -- it's time to get it done
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm
gs_font = fm.FontProperties(
                fname='/System/Library/Fonts/Supplemental/GillSans.ttc')

plt.style.use('./old-style.mplstyle')
# plt.grid()
WIDTH, HEIGHT, DPI = 700, 500, 100
fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
linestyle = ['solid', "dashed"]

for i in range(len(spectrum_arr)):
    ax.plot(SN, total_delta_inter[i], color="k", linestyle=linestyle[i], label=spectrum_names_direct[i])

plt.xlabel("S/N", fontsize=14)
plt.ylabel("Delta", fontsize=14)
# plt.yscale("log")
plt.legend()
plt.show()
