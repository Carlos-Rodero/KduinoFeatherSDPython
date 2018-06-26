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

    def user_input_from_terminal(self):
        """Get path from user's input
        Returns
        -------
            path: str
                user's input path
        """
        user_input = input('Give me a number: ')

    def open_file(self):
        """Open DATA.TXT from user input
        Returns
        -------
            True/False: Bool
                It indicates if the procedure was successful
        """
        try:
            # read file
            logFile = r'path\tofile\fileName.txt'

            # open file in write mode
            report = open(logFile, 'w')
        except Exception as e:
            # get line number and error message
            report.write('an error message')

        finally:
            report.close()
