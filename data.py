import os
import re
from io import StringIO

import matplotlib.pyplot as plt
import mooda
import numpy as np
import pandas as pd


class Data:
    """Class to create a data object from DATA.TXT files
    It contains functions related to manage this data object
    """
    def __init__(self, content):
        """It creates the instance of following variables:
        content -- a string that contains file's content
        """
        # Instance variable
        self.content = content

    def content_to_dataframe(self):
        """Get path from user's input
        Returns
        -------
            path: str
                user's input path
        """
        self.start_string_metadata = r"METADATA"
        self.stop_string_metadata = r"DATA"

        self.start_string_data = r"\bDATA\b"
        self.stop_string_data = r"METADATA"

        self.last_start_string_data = r'\bDATA\b'
        self.end_string_data = r'$(?![\r\n])'

        metadata_patron = re.compile(r'{}(?P<length>)\s*(?P<table>[\s\S]*?){}'.
                                     format(self.start_string_metadata,
                                            self.stop_string_metadata))

        data_patron = re.compile(r'{}(?P<length>)\s*(?P<table>[\s\S]*?){}'.
                                 format(self.start_string_data,
                                        self.stop_string_data))

        end_data_patron = re.compile(r'{}(?P<length>)\s*(?P<table>[\s\S]*?){}'.
                                     format(self.last_start_string_data,
                                            self.end_string_data))

        selected_info = ""
        metadata = {}

        # Regular expression to find the metadata patron
        for m in re.finditer(metadata_patron, self.content):
            column_names = []
            values = []
            selected_info = m.group('table')
            lines = selected_info.splitlines()

            for line in lines:
                key = line.split(":")[0]
                if line.count(":") > 1:
                    date_splitted = (line.rsplit(":")[-3:])
                    date_splitted = " ".join(date_splitted)
                    value = date_splitted
                    metadata[key] = value
                else:
                    value = line.split(":")[1]
                    metadata[key] = value.strip()

        # Regular expression to find the data patron
        '''
        for m in re.finditer(data_patron, self.content):
            selected_info_data = m.group('table')
            print("data_patron" + selected_info_data)
            data = StringIO(selected_info_data)
            # create dataframe for this patron iteration
            df = pd.read_csv(data, skipinitialspace=True, skiprows=1,
                             delimiter=' ')
        '''
        # Regular expression to find the last data patron
        for m in re.finditer(end_data_patron, self.content):
            selected_info_data = m.group('table')
            data = StringIO(selected_info_data)

            # create dataframe for this patron iteration
            df = pd.read_csv(data, skipinitialspace=True, skiprows=1,
                             header=None, delimiter=' ',
                             parse_dates={'TIME': [0, 1]}).set_index('TIME')
            df.columns = range(df.shape[1])

        return((metadata.copy(), df.copy()))

    def to_wf(self, metadata, raw):
        wf = mooda.WaterFrame()
        wf.metadata = metadata
        wf.data['RED'] = raw[0]
        wf.data['GREEN'] = raw[1]
        wf.data['BLUE'] = raw[2]
        wf.data['CLEAR'] = raw[3]

        red = {'units': "counts"}
        wf.meaning['RED'] = red

        for i in range(len(raw.columns)):
            if i < 4:
                continue
            if i % 4 == 0:
                wf.data['RED'] += raw[i]
                wf.data['GREEN'] += raw[i+1]
                wf.data['BLUE'] += raw[i+2]
                wf.data['CLEAR'] += raw[i+3]

        wf.data['RED_QC'] = 0
        wf.data['GREEN_QC'] = 0
        wf.data['BLUE_QC'] = 0
        wf.data['CLEAR_QC'] = 0

        wf.barplot('RED')
        plt.show()
        return wf
