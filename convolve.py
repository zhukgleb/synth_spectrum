from data import extract_data
import matplotlib.pyplot as plt



a1, f1 = extract_data("data/NES_model", text=True)
a2, f2 = extract_data("data/NES_model_110000.rgs", text=True)
a3, f3 = extract_data("data/NES_model_40000.rgs", text=True)
a4, f4 = extract_data("data/NES_model_80000.rgs", text=True)
a5, f5 = extract_data("data/NES_model_5000.rgs", text=True)


plt.plot(a1, f1, label="Ideal")
# plt.plot(a2, f2, label="Resolution: 110000")
# plt.plot(a3, f3, label="Resolution: 40000")
plt.plot(a4, f4, label="Resolution: 80000")
plt.plot(a5, f5, label="Resolution: 5000")
plt.xlim(5000,5100)
plt.legend()
plt.show()
