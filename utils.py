import json



def save_json(path, data):
    with open(path, "w") as write_file:
        json.dump(data, write_file, ensure_ascii=False)
        
        
def load_json(path):
    with open(path, "r") as read_file:
        in_data = json.load(read_file)
        
    return in_data