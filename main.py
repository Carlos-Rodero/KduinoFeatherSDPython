from data import Data


def main():
    d = Data()
    path = d.user_input_from_terminal()
    if d.open_file(path):
        print("obre el fitxer")
    d.convert_to_csv()


if __name__ == "__main__":
    main()
