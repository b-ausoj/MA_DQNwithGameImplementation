# © by Josua Bürki, 2019
# This is the class for the Q-Table
import pickle
from src import functions as f


class QTable:

    def __init__(self, rows, columns, initialize=True, content=None):

        if initialize:
            if isinstance(content, float) or isinstance(content, int):
                self.table = f.create_matrix(rows, columns, content)
            else:
                self.table = f.create_matrix(rows, columns, 0.0)
        else:
            with open("QTable.txt", "rb") as read_file:
                self.table = pickle.load(read_file)

    def save_table(self):
        with open("QTable.txt", "wb") as write_file:
            pickle.dump(self.table, write_file)
