import os
import re
from io import StringIO
import matplotlib.pyplot as plt
import mooda_code.mooda as mooda
import numpy as np
import pandas as pd
import csv
from itertools import combinations


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

        # Initialize index waterframe
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

            wf.resample('S')

            # Delete last index because it is a minute that we are not going to
            # use
            wf.data.drop(wf.data.tail(1).index, inplace=True)

            # Extract data of the dataframe raw
            red_list = []
            green_list = []
            blue_list = []
            clear_list = []
            for j in range(len(raw.index)-1):
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

            wf.data['RED'] = red_array
            wf.data['GREEN'] = green_array
            wf.data['BLUE'] = blue_array
            wf.data['CLEAR'] = clear_array

        wf.data['RED_QC'] = 0
        wf.data['GREEN_QC'] = 0
        wf.data['BLUE_QC'] = 0
        wf.data['CLEAR_QC'] = 0

        return wf

    def horizontal_sensor_analysis(self, waterframes, start_time, stop_time):
        """Analysis of different sensors in horizontal
        Parameters
        ----------
            waterframes: list
                List with waterframes from sensors.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
        Returns
        -------
            wf: WaterFrame object to manage this data series .
        """
        # Concat all waterframes and rename parameters
        wf_all = mooda.WaterFrame()
        names = []
        for wf in waterframes:
            name = wf.metadata["name"]
            names.append(name)
            wf_all.concat(wf)
            for parameter in wf.parameters():
                wf_all.rename(parameter, "{}_{}".format(parameter, name))

            '''individual analysis for each sensor'''
            '''uncomment next lines to do cumulative analysis'''
            # plot timeseries_cumulative
            # d.timeseries_cumulative_plot(wf, name)
            # wf.tsplot(['RED', 'GREEN', 'BLUE', 'CLEAR'], rolling=1)
            # plt.show()

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # hist
        '''
        wf_all.hist(parameter=["CLEAR_14", "CLEAR_15", "CLEAR_17", "CLEAR_18",
                               "CLEAR_19"], mean_line=True)
        plt.show()
        '''
        # create .csv with resampling data
        with open('results_correlation_resample.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            header = ['sensor'] + list(range(1, 60))
            writer.writerow(header)

            df = pd.DataFrame()
            label_index = []
            for i in range(1, 60):
                # copy waterframe to avoid resample the same waterframe in
                # loop
                wf_all_copy = mooda.WaterFrame()
                wf_all_copy.data = wf_all.data.copy()
                wf_all_copy.resample("{}S".format(i))

                # combination of parameters for pairs
                for combo in combinations(wf_all_copy.parameters(), 2):
                    param_name_1 = " ".join(re.findall("[a-zA-Z]+", combo[0]))
                    param_name_2 = " ".join(re.findall("[a-zA-Z]+", combo[1]))
                    if param_name_1 == param_name_2:
                        # print(combo, wf_all_copy.corr(combo[0], combo[1]), i)
                        if i == 1:
                            label_index.append("{}_{}".format(combo[0],
                                                              combo[1]))

                            df = df.append({i: wf_all_copy.corr(combo[0],
                                                                combo[1])},
                                           ignore_index=True)
                        else:
                            df[i] = wf_all_copy.corr(combo[0], combo[1])
            print(df)
            '''
                row = [wf_all_copy.corr("CLEAR_14", "CLEAR_15")]
                writer.writerow([row])
            '''
                # print(wf_all_copy.parameters)

                # print("14 - 15", wf_all_copy.corr("CLEAR_14", "CLEAR_15"))
                # print("14 - 17", wf_all_copy.corr("CLEAR_14", "CLEAR_17"))
                # print("14 - 18", wf_all_copy.corr("CLEAR_14", "CLEAR_18"))
                # print("14 - 19", wf_all_copy.corr("CLEAR_14", "CLEAR_19"))
                # print("15 - 17", wf_all_copy.corr("CLEAR_15", "CLEAR_17"))

        """
        wf_all.resample("10S")
        print("14 - 15", wf_all.corr("CLEAR_14", "CLEAR_15"))
        print("14 - 17", wf_all.corr("CLEAR_14", "CLEAR_17"))
        print("14 - 18", wf_all.corr("CLEAR_14", "CLEAR_18"))
        print("14 - 19", wf_all.corr("CLEAR_14", "CLEAR_19"))
        print("15 - 17", wf_all.corr("CLEAR_15", "CLEAR_17"))
        # print(wf_all.max_diff("CLEAR_14", "CLEAR_19"))

        wf_all.scatter_matrix(keys=["CLEAR_14", "CLEAR_15", "CLEAR_17",
                                    "CLEAR_18", "CLEAR_19"])
        wf_all.tsplot(["CLEAR_14", "CLEAR_15", "CLEAR_17",
                       "CLEAR_18", "CLEAR_19"])

        plt.show()

        """

    def timeseries_cumulative_plot(self, wf, name):
        """Makes plots of time series from waterframe parameter.
        Parameters
        ----------
            wf: WaterFrame
                WaterFrame object to manage this data series.
            name: string
                Name from waterframe metadata
        """

        wf.slice_time('20180822120000', '20180822122500')
        axes = plt.gca()
        axes.set_ylim([0, 300000])
        wf.tsplot(['RED', 'GREEN', 'BLUE', 'CLEAR'], rolling=1, ax=axes)
        plt.title('Figure {}'.format(name))
        plt.show()

    # def concat_all_wf(self, waterframes):
        """Concat all waterframes and rename parameters
        Parameters
        ----------
            waterframes: list
                A list with all waterframes to concat.
        Returns
        -------
            wf_all: WaterFrame object to manage this data series.
        """
        """
        wf_all = mooda.WaterFrame()
        for wf in waterframes:
            name = wf.metadata["name"]
            wf_all.concat(wf)
            for parameter in wf.parameters():
                wf_all.rename(parameter, "{}_{}".format(parameter, name))
        return wf_all
        """
