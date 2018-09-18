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
        """It converts variable content to dataframe
        Returns
        -------
            metadata: dict
                A dictionary that contains the metadata information of the
                content
            df: pandas DataFrame
                A pandas DataFrame that contains the measurement values of
                the content
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

    def to_wf(self, metadata, raw, cumulative=False):
        """It converts metadata and raw data to WaterFrame object.
        Parameters
        ----------
            metadata: dict
                Dictionary with metadata information of the content.
            raw: pandas DataFrame
                A pandas Dataframe that contains the measurement
                values of the timeserie.
            cumulative: boolean, optional (cumulative = False)
                It makes a cumulative dataframe adding data
        Returns
        -------
            wf: WaterFrame object to manage this data series .
        """
        wf = mooda.WaterFrame()
        wf.metadata = metadata
        wf.data['RED'] = raw[0]
        wf.data['GREEN'] = raw[1]
        wf.data['BLUE'] = raw[2]
        wf.data['CLEAR'] = raw[3]

        red = {'units': "counts"}
        wf.meaning['RED'] = red

        green = {'units': "counts"}
        wf.meaning['GREEN'] = green

        blue = {'units': "counts"}
        wf.meaning['BLUE'] = blue

        clear = {'units': "counts"}
        wf.meaning['CLEAR'] = clear

        if cumulative is True:
            for i in range(len(raw.columns)):
                if i < 4:
                    continue
                if i % 4 == 0:
                    wf.data['RED'] += raw[i]
                    wf.data['GREEN'] += raw[i+1]
                    wf.data['BLUE'] += raw[i+2]
                    wf.data['CLEAR'] += raw[i+3]
        else:
            # wf.data = wf.data.resample('T', label='right')
            # wf.data = wf.data.resample('S')
            # wf.data = wf.data.resample('S', label='right')

            # we have to add 1 minute at the end
            # print(wf.data.index)
            # print(wf.data.index[len(wf.data.index)-1])
            red_list = []
            green_list = []
            blue_list = []
            clear_list = []
            for j in range(len(raw.index)):
                for i in range(len(raw.columns)):
                    if i % 4 == 0:
                        red_list.append(raw[i][j])
                        green_list.append(raw[i+1].iloc[j])
                        blue_list.append(raw[i+2].iloc[j])
                        clear_list.append(raw[i+3].iloc[j])
            red_array = np.array(red_list)
            green_array = np.array(green_list)
            blue_array = np.array(blue_list)
            clear_array = np.array(clear_list)
            # print(red_array)
            # print(len(wf.data.index))
            # wf.data['RED'] = red_array
            """
            index_count = 0
            for i in range(len(raw.columns)):
                if i % 4 == 0:
                    print(wf.data.index[0])
                    wf.data['RED'][wf.data.index[i]] = raw[i]
                    wf.data['GREEN'].loc[wf.data.index[i+1]] = raw[i+1]
                    wf.data['BLUE'].loc[wf.data.index[i+2]] = raw[i+2]
                    wf.data['CLEAR'].loc[wf.data.index[i+3]] = raw[i+3]
                    index_count += 4"""

        wf.data['RED_QC'] = 0
        wf.data['GREEN_QC'] = 0
        wf.data['BLUE_QC'] = 0
        wf.data['CLEAR_QC'] = 0

        # print(wf.data.tail())

        return wf

    def timeseries_plot(self, wf):
        """Makes plots of time series from waterframe parameter.
        Parameters
        ----------
            wf: WaterFrame
                WaterFrame object to manage this data series.
        """

        wf.slice_time('20180822120000', '20180822122500')
        axes = plt.gca()
        axes.set_ylim([0, 300000])
        wf.tsplot(['RED', 'GREEN', 'BLUE', 'CLEAR'], rolling=1, ax=axes)
        plt.title('Figure x')
        plt.show()
