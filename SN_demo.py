from data import extract_data
from velocity import find_velocity
import PyAstronomy.pyasl as pyasl
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile
import astropy.units as u


#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 22} 
#import matplotlib
#matplotlib.rc('font', **font)

def make_good():
    cool = extract_data("data/NES_model_110000.rgs", text=True)
    lbv = extract_data("data/MODEL1_NOSHIFT.data", text=True)
    a_template, f_template = extract_data("data/NES_model_110000.rgs", text=True)

#     a_template, f_template = extract_data("data/MODEL1_NOSHIFT.data", text=True)

    spectrum_arr = [cool]

    spectrum_names = ["R=60000 inter", "R=15000 inter"]
    spectrum_names_direct = ["R=60000", "R=40000", "R=15000"]

    total_velocity_data = []
    total_delta = []
    total_delta_inter = []
    total_velocity_err = []

    v = 20 * 10 # in meters
    dots = 10
    plot = False

    for i in range(len(spectrum_arr)):
        velocity = []
        z_velocity = []
        SN = []
        delta = []
        delta_inter = []
        z_err_arr = []
        # Now, make a variance between arrays -- add some noise
        # from SN 1 to 100
        for j in range(100, 101, 1):
            print(f"SN is {j}")
            ang = np.copy(spectrum_arr[i][0])
            flux = np.copy(spectrum_arr[i][1])
            _, ang = pyasl.dopplerShift(ang, flux, v / 1000, edgeHandling="firstlast")
            noise_spectrum = np.copy(flux)
            noise = np.random.normal(loc=0, scale=1/j, size=len(flux))
            # noise = 0
            noise_spectrum = noise_spectrum + noise
            cv, z, z_err, s = find_velocity([ang, noise_spectrum], 
                                            [a_template, f_template],
                                            [5200, 7700], dots)
            velocity.append(cv)
            z_velocity.append(z)
            SN.append(j)
            delta_inter.append(v - z)  # For delta graph
            delta.append(v - cv)
            z_err_arr.append(s)

            del noise_spectrum
            del ang
            del flux

        velocity_data = [velocity, z_velocity]
        total_velocity_data.append(velocity_data)
        total_delta_inter.append(delta_inter)
        total_delta.append(delta)
        total_velocity_err.append(z_err_arr)
        

# A very bad part. btw -- it's time to get it done
    from matplotlib.ticker import MultipleLocator
    import matplotlib.font_manager as fm
#    gs_font = fm.FontProperties(
#                    fname='/System/Library/Fonts/Supplemental/GillSans.ttc')

    plt.style.use('./old-style.mplstyle')
    WIDTH, HEIGHT, DPI = 700, 500, 150
    fig, ax = plt.subplots(figsize=(WIDTH/DPI, HEIGHT/DPI), dpi=DPI)
    linestyle = ['solid', "dashed", 'dotted', 'dashdot', 'solid', 'dashed','dotted', 'dashdot', 'solid', 'dashed']

    for i in range(len(spectrum_arr)):
#        ax.errorbar(SN, total_delta[i], total_velocity_err[i], color="k", linestyle=linestyle[i], label=spectrum_names_direct[i])
        plt.plot(SN, total_delta[i], color="k", linestyle=linestyle[i], label=spectrum_names_direct[i])


    if plot:
        plt.title(f"Delta graph for {v} m/s")
        plt.xlabel("S/N", fontsize=14)
        plt.ylabel("Delta", fontsize=14)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    make_good()
