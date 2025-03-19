from Dynaconf import Dynaconf

def load_configs():
    return Dynaconf(settings_file=["../../conf/configs.toml"])
