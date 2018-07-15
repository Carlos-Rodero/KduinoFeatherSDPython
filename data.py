import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Data:
    """Class to create a data object from DATA.TXT files
    It contains functions related to manage this data object
    """
    def __init__(self, content):
        """Adding new attributes:
        data -- A pandas DataFrame that contains the measurement values of the
        time series.
        metadata -- A dictionary that contains the metadata information of the
        time series.
        """
        # Attributes
        self.__content = content
        self.data = pd.DataFrame()
        self.metadata = dict()

    @property
    def content(self):
        self.start_string_metadata = r"METADATA"
        self.stop_string_metadata = r"DATA"

        self.start_string_data = r"DATA"
        self.stop_string_data = r"METADATA"

        patron_metadata = re.compile(r'{}(?P<length>)\s*(?P<table>[\s\S]*?){}'.format(self.start_string_metadata, self.stop_string_metadata))

        patron_data = re.compile(r'{}\s*(?P<length>\d+\.\d+)\s*nm\s*(?P<table>[\s\S]*?){}'.format
                                 (self.start_string_data,
                                  self.stop_string_data))

        selected_info = ""

        # Creation of pandas dataframe with useful data
        # df_final = pd.DataFrame()
        # output_list = []

        # Regular expression to find the patron
        for m in re.finditer(patron_metadata, self.__content):
            selected_info = m.group('table')
            print(selected_info)

            # data = StringIO(selected_info)

            # create dataframe for this patron iteration
            # df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
        

    

