import json

class Action:
    def __init__(self) -> None:
        self.action = []
        try:
            with open("data_preparation/actions.json", 'r') as f:
                data = json.load(f)
            self.action = data['all_actions']
        except FileNotFoundError:
            self.action = []

    def append_to_actions_list(self, name) :
        if (name not in self.action):
            self.action.append(name)
        with open("data_preparation/actions.json", 'w') as f:
            json.dump({"all_actions": self.action}, f)