import json
import json
import urllib
import requests

class Action:
    def __init__(self) -> None:
        self.action = []

        json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
        json_filename = "data_preparation/kinetics_classnames.json"

        response = requests.get(json_url)
        with open(json_filename, 'wb') as file:
            file.write(response.content)

        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)

        self.kinetics_classnames = kinetics_classnames

        sorted_kinetics = sorted(self.kinetics_classnames.items(), key=lambda x: x[1], reverse=False)

        for action in sorted_kinetics:
            self.action.append(action[0].replace('"', ""))



        try:
            with open("data_preparation/actions.json", 'r') as f:
                data = json.load(f)
            for action in data['all_actions']:
                self.action.append(action)
        except FileNotFoundError:
            print("File not found")

        
            

    def append_to_actions_list(self, name) :
        if (name not in self.action):
            self.action.append(name)
        with open("data_preparation/actions.json", 'w') as f:
            json.dump({"all_actions": self.action}, f)



Action()