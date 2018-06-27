import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Data:
    """Class to manage Kduino Feather SD
    It contains functions related to extract information from user's input.
    """
    def __init__(self):
        """Adding new attributes:
        data -- A pandas DataFrame that contains the measurement values of the
        time series.
        metadata -- A dictionary that contains the metadata information of the
        time series.
        """
        # Attributes
        self.data = pd.DataFrame()
        self.metadata = dict()
        self.content_list = {}

    def user_input_from_terminal(self):
        """Get path from user's input
        Returns
        -------
            path: str
                user's input path
        """
        path = input("Enter path of your DATA.TXT (press enter to " +
                     "set default path): ")
        if path == "":
            path = os.path.join(os.getcwd(), 'DATA.TXT')
        return path

    def open_file(self, path):
        """Open DATA.TXT from user input
        Args
        ----
            path: str
                filename to read
        Returns
        -------
            True/False: Bool
                It indicates if the procedure was successful
        Raises
        ------
            ValueError: Unable to read file
        """
        try:
            data_file = open(path)
            self.content_list = data_file.readlines()
            return True
        except IOError:
            print("The file does not exist")
            return False

    def convert_to_csv(self):
        """convert list to a .csv file
        Returns
        -------
            True/False: Bool
                It indicates if the procedure was successful
        """
        for element in self.content_list:
            element.strip('\n')
        print(self.content_list)
