Data Analysis

Cumulative data: sum all counts in period of time
Non cumulative data: all counts separately in period of time

Functions:

timeseries: plot
hist: plot
max diff: plot, csv
scatter matrix: plot
correlation resample: csv

Results:

    safata_Stirling/cumulative:
        - timeseries

    safata_Stirling/non_cumulative:
        - timeseries
        - histogram
        - max_diff
        - scatter_matrix

    safata_loch_Leven/cumulative:
        - timeseries
    
    safata_loch_Leven/non_cumulative:
        - timeseries
        - histogram
        - max_diff
        - scatter_matrix

