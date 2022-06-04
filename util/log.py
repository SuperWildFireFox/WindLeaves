import io
import json
import os
import sys


class Logger(object):
    def __init__(self, filename, path="./", mode='w'):
        self.log = open(os.path.join(path, filename), mode=mode, encoding='utf-8')

    def write(self, message, print_=True):
        if print_:
            print(message)
        self.log.write(message+"\n")
        self.log.flush()

    def __del__(self):
        self.log.close()
