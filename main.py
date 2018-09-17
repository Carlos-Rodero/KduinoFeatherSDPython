from utility import Utility
from data import Data
from mooda import WaterFrame
import matplotlib.pyplot as plt


def main():
    path = Utility.user_input_from_terminal()
    contents = Utility.open_files(path)
    for content in contents:
        d = Data(content)
        metadata, raw = d.content_to_dataframe()
        wf = d.to_wf(metadata, raw, cumulative=False)
        # d.timeseries_plot(wf)
        exit()


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
