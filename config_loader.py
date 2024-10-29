import configparser

config = configparser.ConfigParser()
config.read("settings.ini")
tsfit_path = config["TSFit"]["tsfit_path"]
