from data import extract_data
import matplotlib.pyplot as plt
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl


s2 = extract_data("data/NES_model_110000.rgs", text=True)
s3 = extract_data("data/NES_model_60000.rgs", text=True)
s4 = extract_data("data/NES_model_40000.rgs", text=True)
s5 = extract_data("data/NES_model_15000.rgs", text=True)
s6 = extract_data("data/NES_model_30000.rgs", text=True)
a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

spectrum_arr = [s2, s3, s4, s5]
spectrum_names = ["R=110000 inter", "R=60000 inter", "R=40000 inter", "R=15000 inter"]
spectrum_names_direct = ["R=110000", "R=60000", "R=40000", "R=15000"]
# spectrum_arr = [s5]
# spectrum_names = ["R=15000 inter"]
# spectrum_names_direct = ["R=15000"]

total_velocity_data = []
for i in range(len(spectrum_arr)):
    velocity = []
    z_velocity = []
    true_velocity = []
    for j in range(1, 100):
       print(f"speed is {j} meters")
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], -j/1000, edgeHandling="firstlast")
       mult = 100
       cv, z = find_velocity(spectrum_arr[i], [a_template, f_template], [5000, 5100], mult)
       velocity.append(cv)
       z_velocity.append(z)
       true_velocity.append(-j)
       _, spectrum_arr[i][0] = pyasl.dopplerShift(spectrum_arr[i][0], spectrum_arr[i][1], j/1000, edgeHandling="firstlast")
    velocity_data = [true_velocity, velocity, z_velocity]
    total_velocity_data.append(velocity_data)


# A very bad part. btw -- it's time to get it done
plt.style.use('seaborn-v0_8-talk')
plt.grid()
for i in range(len(spectrum_arr)):
    plt.scatter(total_velocity_data[i][0], total_velocity_data[i][2], label=spectrum_names[i], alpha=0.7)
#     plt.scatter(total_velocity_data[i][0], total_velocity_data[i][1], label=spectrum_names_direct[i], marker="+", alpha=0.9)

plt.xlabel("Input speed, m/s")
plt.ylabel("Output speed, m/s")
plt.legend()
plt.plot([0, -100], [0, -100], linestyle='dashed', color="gray", alpha=0.5)
plt.show()
