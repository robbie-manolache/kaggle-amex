import os
import json

def env_config(config_file: str, 
               gen_template: bool = False):
    """
    Sets environment variables by reading names and values from a JSON file.
    ***
    ARGS
    config_file: str, path to JSON file with configuration settings
    ***
    """
    with open(config_file) as rf:
        config = json.load(rf)
    
    config_template = {}
    for k, v in config.items():
        os.environ[k] = v
        config_template[k] = ""
        print("Value for %s has been set!"%(k))

    if gen_template:
        with open("config_template.json", "w") as wf:
            json.dump(config_template, wf)
        