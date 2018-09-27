from utility import Utility
from data import Data
# from mooda import WaterFrame
# import matplotlib.pyplot as plt


def main():
    # select path where are DATA.TXT
    path = Utility.user_input_from_terminal()
    # obtain list with DATA.TXT content file's
    contents = Utility.open_files(path)
    # set analysis cumulative or not
    cumulative = False

    waterframes = []
    for content in contents:
        # initialize Data object
        d = Data(content)
        # obtain dataframe and metadata from content
        metadata, raw = d.content_to_dataframe()
        # obtain waterframe. Data could be cumulative (cumulative=True)
        wf = d.to_wf(metadata, raw, cumulative=cumulative)
        # add waterframe to waterframes list
        waterframes.append(wf)

    def loch_Leven_Tray():
        start_time = '20180822120000'
        stop_time = '20180822122500'

        # timeseries plot
        """ d.timeseries_plot(waterframes, start_time, stop_time,
                        cumulative=cumulative) """
        # hist plot
        """ d.hist_plot(waterframes, start_time, stop_time,
                        cumulative=cumulative) """

        # max diff
        """ d.max_diff_sensors(waterframes, start_time, stop_time,
                           cumulative=cumulative) """

        # scatter matrix
        """ d.scatter_matrix(waterframes, start_time, stop_time,
                         cumulative=cumulative) """

        # correlation resample
        """ d.correlation_resample(waterframes, start_time, stop_time,
                               cumulative=cumulative) """

    def loch_Leven_Buoy():
        start_time = ''
        stop_time = ''

    """ Analysis Loch Leven Tray """
    loch_Leven_Tray()

    """ Analysis Loch Leven Buoy """
    loch_Leven_Buoy()


if __name__ == "__main__":
    main()
