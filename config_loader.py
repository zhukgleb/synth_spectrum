import configparser

config = configparser.ConfigParser()
config.read("settings.ini")
tsfit_path = config["TSFit"]["tsfit_path"]
tsfit_output = config["TSFit"]["tsfit_output"]


def tsfit_pathes():
    return {"ts_out": tsfit_output}
