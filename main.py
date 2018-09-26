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
    cumulative = True

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

    '''LOCH LEVEN TRAY'''
    start_time = '20180822120000'
    stop_time = '20180822122500'
    # timeseries plot
    """ d.timeseries_plot(waterframes, start_time, stop_time,
                      cumulative=cumulative) """
    # hist plot
    # treure nomes el clear
    d.hist_plot(waterframes, start_time, stop_time, cumulative=cumulative)

    # horizontal sensor analysis
    # d.horizontal_sensor_analysis(waterframes, '20180822121000',
                                 # '20180822122500')


def comparacio_sensors():
    path14 = r"C:\Users\caroga\Google Drive\Monocle\github\
    ProcessarDadesKdUINOFeather\safata_loch_Leven\DATA_14.TXT"
    path18 = r"C:\Users\caroga\Google Drive\Monocle\github\
    ProcessarDadesKdUINOFeather\safata_loch_Leven\DATA_18.TXT"
    path19 = r"C:\Users\caroga\Google Drive\Monocle\github\
    ProcessarDadesKdUINOFeather\safata_loch_Leven\DATA_19.TXT"

    f = open(path14, "r")
    content = f.read()
    d = Data(content)
    metadata, raw = d.content_to_dataframe()
    wf14 = d.to_wf(metadata, raw)

    f = open(path18, "r")
    content = f.read()
    d = Data(content)
    metadata, raw = d.content_to_dataframe()
    wf18 = d.to_wf(metadata, raw)

    f = open(path19, "r")
    content = f.read()
    d = Data(content)
    metadata, raw = d.content_to_dataframe()
    wf19 = d.to_wf(metadata, raw)

    wf_all = WaterFrame()
    wf_all.concat(wf14)
    for parameter in wf14.parameters():
        wf_all.rename(parameter, "{}_S14".format(parameter))

    wf_all.concat(wf18)
    for parameter in wf18.parameters():
        wf_all.rename(parameter, "{}_S18".format(parameter))

    wf_all.concat(wf19)
    for parameter in wf19.parameters():
        wf_all.rename(parameter, "{}_S19".format(parameter))

    wf_all.slice_time('20180822120000', '20180822122500')

    print(wf_all.corr("CLEAR_S14", "CLEAR_S19"))
    print(wf_all.max_diff("CLEAR_S14", "CLEAR_S19"))

    wf_all.scatter_matrix(keys=["CLEAR_S14", "CLEAR_S19", "CLEAR_S18"])

    plt.show()


if __name__ == "__main__":
    main()
    # comparacio_sensors()
