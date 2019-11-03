import os

def check_dir_exists(path):
    return os.path.exists(path)

def check_is_file(path):
    return os.path.isfile(path)

def check_file_exists(path):
    return os.path.exists(path)

def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        return False
    else:
        print ("Successfully created the directory %s " % path)
        return True

def delete_file(path):
    try:
        os.remove(path)
    except OSError:
        print ("Deleting file %s failed" % path)
        return False
    else:
        print ("Creating file %s " % path)
        return True