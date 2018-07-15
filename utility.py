import os
import glob


class Utility:
    """Class to manage Kduino Feather SD
    It contains functions related to extract information from user's input.
    """

    @staticmethod
    def user_input_from_terminal():
        """Get path from user's input
        Returns
        -------
            path: str
                user's input path
        """
        path = input("Enter path of folder 'data' where are the DATA.TXT " +
                     "(press enter to set default path): ")
        if path == "":
            path = os.path.join(os.getcwd(), 'data')
        return path

    @staticmethod
    def open_files(path):
        """Open files DATA.TXT
        Args
        ----
            path: str
                folder where are DATA.TXT files
        Returns
        -------
            contents: list
                list with DATA.TXT content file's
        Raises
        ------
            IOError: Unable to read file
        """
        contents = []
        for filename in glob.glob(os.path.join(path, '*.TXT')):
            content = {}
            try:
                f = open(filename, "r")
                print("open filename: " + filename)
                # content = f.read().splitlines()
                content = f.read()
                contents.append(content)
            except IOError:
                print("The file does not exist")
        return contents

    '''def convert_to_csv(self):
        """convert list to a .csv file
        Returns
        -------
            True/False: Bool
                It indicates if the procedure was successful
        """
        for element in self.content_list:
            element.strip('\n')
        print(self.content_list)
        '''
