from data import extract_data
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof


# s2 = extract_data("data/NES_model_110000.rgs", text=True)
s3 = extract_data("data/NES_model_60000.rgs", text=True)
# s4 = extract_data("data/NES_model_40000.rgs", text=True)
s5 = extract_data("data/NES_model_15000.rgs", text=True)
a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

spectrum_arr = [s3]
# spectrum_arr = [[a_template, f_template]]
spectrum_names = ["R=60000 inter", "R=15000 inter"]
spectrum_names_direct = ["R=60000","R=15000"]
# cv, z = find_velocity([a_template, f_template], [a_template, f_template],
#                              [4600, 5400], 50)

total_velocity_data = []
total_delta = []
total_delta_inter = []

v = 20*1000 # in meters
dots = 100

for i in range(len(spectrum_arr)):
    velocity = []
    z_velocity = []
    SN = []
    delta = []
    delta_inter = []
    z_err_arr = []
    # Now, make a variance between arrays -- add some noise
    # from SN 1 to 100

    for j in range(300, 301, 1):
       print(f"SN is {j}")
       ang = np.copy(spectrum_arr[i][0])
       flux = np.copy(spectrum_arr[i][1])
#       print(round(getsizeof(flux) / 1024 / 1024,2))

       _, ang = pyasl.dopplerShift(ang, flux, v/1000, edgeHandling="firstlast")
       noise_spectrum = np.copy(flux)
       noise = np.random.normal(loc=0, scale=1/j, size=len(flux))
       noise_spectrum = noise_spectrum + noise
       cv, z, z_err = find_velocity([ang, noise_spectrum], [a_template, f_template],
                             [4500, 5000], dots)
       velocity.append(cv)
       z_velocity.append(z)
       SN.append(j)
       delta_inter.append(v - z)  # For delta graph
       delta.append(v - cv)
       z_err_arr.append(z_err)

       del noise_spectrum
       del ang
       del flux
 

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
    ax.errorbar(SN, total_delta_inter[i], z_err_arr[i], color="k", linestyle=linestyle[i], label=spectrum_names_direct[i])

plt.title(f"Delta graph for {v} m/s")
plt.xlabel("S/N", fontsize=14)
plt.ylabel("Delta", fontsize=14)
# plt.yscale("log")
plt.legend()
plt.show()
