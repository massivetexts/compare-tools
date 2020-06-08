import os

def already_imported_list(dir):
    already_completed = set()

    for file in os.listdir(dir):
        if "already_completed_files" in file:
            lines = open("{}/{}".format(dir, file)).readlines()
            for line in lines:
                already_completed.add(line.rstrip())
                
    return already_completed