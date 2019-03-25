import pickle


class Logger():
    def __init__(self):
        self.logs = {}

    def __setitem__(self, key, item):
        """
        Maintains a list of the values passed to it
        :param key:
        :param item:
        :return:
        """
        if key not in self.logs:
            self.logs[key] = []
        self.logs[key].append(item)

    def __getitem__(self, key):
        """
        Returns the list
        :param key:
        :return:
        """
        return self.logs[key]

    def set_dict(self, data):
        for key, value in data.items():
            self[key] = value

    def get_average(self, key):
        return sum(self.logs[key]) / len(self.logs[key])

    def get_averages(self):
        averages = {}
        for key, val in self.logs.items():
            averages[key] = self.get_average(key)
        return averages

    def reset(self):
        self.logs = {}

    def save(self, path):
        pickle.dump(self.logs, open(path, "wb"))

    def load(self, path):
        self.logs = pickle.load(open(path, "rb"))