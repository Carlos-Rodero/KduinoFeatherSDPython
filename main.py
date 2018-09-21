from utility import Utility
from data import Data
from mooda import WaterFrame
import matplotlib.pyplot as plt
import csv


def main():
    # select path where are DATA.TXT
    path = Utility.user_input_from_terminal()
    # obtain list with DATA.TXT content file's
    contents = Utility.open_files(path)

    waterframes = []
    for content in contents:
        # initialize Data object
        d = Data(content)
        # obtain dataframe and metadata from content
        metadata, raw = d.content_to_dataframe()
        # obtain waterframe. Data could be cumulative (cumulative=True)
        wf = d.to_wf(metadata, raw, cumulative=False)
        # add waterframe to waterframes list
        waterframes.append(wf)

    def horizontal_sensor_analysis(start_time, stop_time):
        # Concat all waterframes and rename parameters
        # wf_all = d.concat_all_wf(waterframes)
        wf_all = WaterFrame()
        names = []
        for wf in waterframes:
            name = wf.metadata["name"]
            names.append(name)
            wf_all.concat(wf)
            for parameter in wf.parameters():
                wf_all.rename(parameter, "{}_{}".format(parameter, name))

            # individual analysis for each sensor

            # plot timeseries_cumulative
            # d.timeseries_cumulative_plot(wf, name)
            # wf.tsplot(['RED', 'GREEN', 'BLUE', 'CLEAR'], rolling=1)
            # plt.show()

        # slice time
        wf_all.slice_time(start_time, stop_time)

        # create .csv with resampling data
        with open('results_correlation_resample.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            header = ['sensor'] + list(range(1, 60))
            writer.writerow(header)

            for i in range(1, 60):
                # copy waterframe to avoid resample the same waterframe in
                # loop
                wf_all_copy = WaterFrame()
                wf_all_copy.data = wf_all.data.copy()
                wf_all_copy.resample("{}S".format(i))

                # fer totes les correlacions entre tots els sensors
                # buscar per escriure csv en iteració

                row = [wf_all_copy.corr("CLEAR_14", "CLEAR_15")]
                writer.writerow([row])

                print("14 - 15 ", wf_all_copy.corr("CLEAR_14", "CLEAR_15"))
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

    '''LOCH LEVEN TRAY'''
    # horizontal sensor analysis
    horizontal_sensor_analysis('20180822121000', '20180822122500')


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
