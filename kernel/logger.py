# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:02:35 2015

@author: voss_ar
"""
import sys
class logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, "a") as log:
            log.write(message)