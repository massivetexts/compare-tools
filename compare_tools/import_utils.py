import os

def already_imported_list():
    already_completed = set()

    for file in os.listdir("data_outputs/"):
        if "already_completed_files" in file:
            lines = open("data_outputs/{}".format(file)).readlines()
            for line in lines:
                already_completed.add(line.rstrip())
                
    return already_completed