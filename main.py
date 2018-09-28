from utility import Utility
from data import Data


def main():
    # select path where are DATA.TXT
    path = Utility.user_input_from_terminal()
    # obtain list with DATA.TXT content file's
    contents = Utility.open_files(path)

    if not contents:
        print("empty data")
        exit()

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

    def analysis_tray(start_time, stop_time):
        # timeseries plot
        d.timeseries_plot(waterframes, start_time, stop_time,
                          cumulative=cumulative)
        # hist plot
        d.hist_plot(waterframes, start_time, stop_time,
                    cumulative=cumulative)

        # max diff
        d.max_diff_sensors(waterframes, start_time, stop_time,
                           cumulative=cumulative)

        # scatter matrix
        d.scatter_matrix(waterframes, start_time, stop_time,
                         cumulative=cumulative)

        # correlation resample
        d.correlation_resample(waterframes, start_time, stop_time,
                               cumulative=cumulative)

    def analysis_buoy(start_time, stop_time):
        # timeseries plot
        """ d.timeseries_plot(waterframes, start_time, stop_time,
                          cumulative=cumulative) """
        # hist plot
        d.kd_plot(waterframes, start_time, stop_time,
                  cumulative=cumulative)

    """ Analysis Loch Leven Tray """
    # analysis_tray('20180822120000', '20180822122500')

    """ Analysis Stirling Tray """
    # analysis_tray('20180821133200', '20180821134500')

    """ Analysis Loch Leven Buoy """
    analysis_buoy('20180822115000', '20180822150000')


if __name__ == "__main__":
    main()
