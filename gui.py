import dearpygui.dearpygui as dpg
from data import extract_data
from math import sin
dpg.create_context()

x, y, _ = extract_data()
res_x, res_y = 1920, 1080
CURRENT_DPI = 150

with dpg.font_registry():
    # first argument ids the path to the .ttf or .otf file
    default_font = dpg.add_font("/usr/share/fonts/OTF/CodeNewRomanNerdFont-Bold.otf", 20)

with dpg.window(label="Spectra", width=int(res_x*0.8), height=int(res_y * 0.8)):
    # themes part
    dpg.bind_font(default_font)
    with dpg.theme(tag="spectrum_theme_1"):
        with dpg.theme_component(0):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (230, 0, 230), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (230, 0, 230, 170), category=dpg.mvThemeCat_Plots)

    with dpg.plot(label="Spectrum plot", height=int(res_y*0.8), width=-1):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="x")
        dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="yaxis")

        dpg.add_shade_series(list(x), list(y), label="Synthetic spectra", parent="yaxis", tag="syn_spectrum")

        # apply theme
        dpg.bind_item_theme(dpg.last_item(), "spectrum_theme_1")


dpg.create_viewport(title='Extractor', width=int(res_x*0.9), height=int(res_y*0.9))
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
