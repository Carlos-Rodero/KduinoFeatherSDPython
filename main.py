from utility import Utility
from data import Data


def main():
    path = Utility.user_input_from_terminal()
    contents = Utility.open_files(path)
    for content in contents:
        d = Data(content)
        metadata, raw = d.content_to_dataframe()
        d.to_wf(metadata, raw)
        exit()


if __name__ == "__main__":
    main()
