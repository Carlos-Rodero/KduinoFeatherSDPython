import os
import re
from io import StringIO

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
        self.content = content
        self.data = pd.DataFrame()
        self.metadata = pd.DataFrame()

    def content_to_dataframe(self):
        self.start_string_metadata = r"METADATA"
        self.stop_string_metadata = r"DATA"

        self.start_string_data = r"DATA"
        self.stop_string_data = r"METADATA"

        metadata_patron = re.compile(r'{}(?P<length>)\s*(?P<table>[\s\S]*?){}'.format(self.start_string_metadata, self.stop_string_metadata))
        data_patron = re.compile(r'{}(?P<length>)\s*(?P<table>[\s\S]*?){}'.format(self.start_string_data, self.stop_string_data))

        selected_info = ""

        # Regular expression to find the metadata patron
        for m in re.finditer(metadata_patron, self.content):
            column_names = []
            values = []
            selected_info = m.group('table')
            lines = selected_info.splitlines()

            for line in lines:
                column_names.append(line.split(":")[0])
                '''if line.count(":") > 1:
                    print("data")
                    values.append(line.rsplit(":")[-1:])
                else:
                    values.append(line.split(":")[1])
                '''

            data = StringIO(selected_info)
            # create dataframe for this patron iteration
            
            df = pd.read_csv(data, sep=':', skiprows=1, names=column_names, error_bad_lines=False, header=None)
            print(df)
            # self.metadata.append(df, ignore_index=True)
            
            # print(self.metadata)

        '''
        # Regular expression to find the data patron
        for m in re.finditer(data_patron, self.content):
            selected_info = m.group('table')
            # print(selected_info)
            # data = StringIO(selected_info)

            # create dataframe for this patron iteration
            # df = pd.read_csv(data, skipinitialspace=True, delimiter=' ')
        '''

        


    
        

    

