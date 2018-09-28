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

    def timeseries_plot(self, waterframes, start_time, stop_time, cumulative):
        """Makes plots of time series from waterframe parameter.
        Parameters
        ----------
            waterframes: list
                List of waterFrame objects to manage this data series.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
            cumulative: boolean, optional (cumulative = False)
                It comes from a cumulative dataframe
        """
        for wf in waterframes:
            name = wf.metadata["name"]
            # plot timeseries_cumulative
            if cumulative:
                wf.slice_time(start_time, stop_time)
                axes = plt.gca()
                axes.set_ylim([0, 300000])
                wf.tsplot(['RED', 'GREEN', 'BLUE', 'CLEAR'], rolling=1,
                          ax=axes)
                plt.title('Figure {}'.format(name))
                plt.show()
            else:
                wf.slice_time(start_time, stop_time)
                wf.tsplot(['RED', 'GREEN', 'BLUE', 'CLEAR'], rolling=1)
                plt.title('Figure {}'.format(name))
                plt.show()

    def hist_plot(self, waterframes, start_time, stop_time, cumulative):
        """Makes plots of histogram from waterframe parameter.
        Parameters
        ----------
            waterframes: list
                List of waterFrame objects to manage this data series.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
            cumulative: boolean, optional (cumulative = False)
                It comes from a cumulative dataframe
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

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # plot histogram
        match_CLEAR = [s for s in wf_all.parameters() if "CLEAR" in s]
        wf_all.hist(parameter=match_CLEAR, mean_line=True)
        plt.show()

    def max_diff_sensors(self, waterframes, start_time, stop_time, cumulative):
        """Show maximum difference between parameters in .csv file
        Parameters
        ----------
            waterframes: list
                List of waterFrame objects to manage this data series.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
            cumulative: boolean, optional (cumulative = False)
                It comes from a cumulative dataframe
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

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # create .csv with sensor's name, timestamp of maximum difference and
        # value of this difference
        with open('results_max_diff.csv', 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)

            for combo in combinations(wf_all.parameters(), 2):
                param_name_1 = " ".join(re.findall("[a-zA-Z]+", combo[0]))
                param_name_2 = " ".join(re.findall("[a-zA-Z]+", combo[1]))
                # if name of parameters are the same (i.e. CLEAR == CLEAR)
                if param_name_1 == param_name_2:
                    where, value = wf_all.max_diff(combo[0], combo[1])
                    filewriter.writerow(["{}_{}".format(combo[0],
                                        combo[1]), where, value])

    def scatter_matrix(self, waterframes, start_time, stop_time,
                       cumulative):
        """Makes scatter matrix plot from waterframe parameters
        Parameters
        ----------
            waterframes: list
                List of waterFrame objects to manage this data series.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
            cumulative: boolean, optional (cumulative = False)
                It comes from a cumulative dataframe
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

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # create scatter matrix plot from CLEAR parameter between different
        # sensors
        match_CLEAR = [s for s in wf_all.parameters() if "CLEAR" in s]
        wf_all.scatter_matrix(keys=match_CLEAR)
        plt.show()

    def correlation_resample(self, waterframes, start_time, stop_time,
                             cumulative):
        """Analysis of correlation between sensors doing different resamples
        Parameters
        ----------
            waterframes: list
                List with waterframes from sensors.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
            cumulative: boolean, optional (cumulative = False)
                It comes from a cumulative dataframe
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

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # create dataframe to convert to csv with resampling data
        df = pd.DataFrame()
        label_index = []
        range_list = range(1, 60)
        first_number_range = range_list[:1]

        for i in range_list:
            # copy waterframe to avoid resample the same waterframe in
            # loop
            wf_all_copy = mooda.WaterFrame()
            wf_all_copy.data = wf_all.data.copy()
            wf_all_copy.resample("{}S".format(i))
            comb_num = 0

            # combination of parameters for pairs
            for combo in combinations(wf_all_copy.parameters(), 2):
                param_name_1 = " ".join(re.findall("[a-zA-Z]+", combo[0]))
                param_name_2 = " ".join(re.findall("[a-zA-Z]+", combo[1]))
                # if name of parameters are the same (i.e. CLEAR == CLEAR)
                if param_name_1 == param_name_2:
                    # print(wf_all_copy.corr(combo[0], combo[1]), i)
                    # the first case of resample, to fill the first column
                    # with expected combination of correlations
                    if i == list(first_number_range)[0]:
                        label_index.append("{}_{}".format(combo[0],
                                                          combo[1]))

                        df = df.append({i: wf_all_copy.corr(combo[0],
                                                            combo[1])},
                                       ignore_index=True)
                    # next columns of dataframe with next resamples
                    else:
                        df.loc[[comb_num], i] = wf_all_copy.corr(combo[0],
                                                                 combo[1])
                        comb_num += 1

        df.insert(0, 'sensors', label_index)
        df.set_index('sensors')
        df.to_csv('results_correlation_resample.csv', sep=' ',
                  encoding='utf-8')

    def kd_plot(self, waterframes, start_time, stop_time, cumulative):
        """Makes Kd plot from histogram average data of all sensors in a buoy.
        Parameters
        ----------
            waterframes: list
                List of waterFrame objects to manage this data series.
            start_time: str
                String about start time to slice.
            stop_time: str
                String about stop time to slice.
            cumulative: boolean, optional (cumulative = False)
                It comes from a cumulative dataframe
        """
        # Concat all waterframes and rename parameters
        wf_all = mooda.WaterFrame()
        names = []
        depths = []
        for wf in waterframes:
            name = wf.metadata["name"]
            names.append(name)
            depth = wf.metadata["depth"]
            depths.append(depth)
            wf_all.concat(wf)
            for parameter in wf.parameters():
                wf_all.rename(parameter, "{}_{}".format(parameter, name))

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # get mean from parameters
        match_CLEAR = [s for s in wf_all.parameters() if "CLEAR" in s]
        means = wf_all.mean(parameter=match_CLEAR).tolist()

        print(means)
        print(np.log(means))
        print(depths)
