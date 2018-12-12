import numpy as np

class Trainer:
    def __init__(self, train_num):
        self.train_num = train_num

    def train(self, train_list:list, test_index:int):
        return

    def verify(self, test_list:list)->float:
        return float()
        
class CrossValidationMaker:
    def __init__(self, split_size:int, data_list:list, trainer:Trainer):
        self.split_size = split_size
        self.validation_data_list = np.array_split(data_list, split_size)
        self.trainer = trainer

    def make_train_test_data(self, test_data_index):
        data_dict = {}
        data_dict['train'] = []
        for index, validation_data in enumerate(self.validation_data_list):
            if index == test_data_index:
                data_dict['test'] = validation_data
            else:
                data_dict['train'].extend(validation_data)

        return data_dict

    def cross_validation(self):
        print('start cross validation ...')
        accuracy_list = [None]*self.split_size
        for test_index in range(self.split_size):
            print('{0}/{1} test ...'.format(test_index, self.split_size))
            data_dict = self.make_train_test_data(test_index)
            self.trainer.train(data_dict['train'], test_index)
            accuracy_list[test_index] = self.trainer.verify(data_dict['test'])
        print('complete cross validation')
        return accuracy_list

def main():
    class TestTrainer(Trainer):
        def __init__(self, train_num):
            super(TestTrainer, self).__init__(train_num)

        def train(self, train_list:list)->list:
            return list()

        def verify(self, test_list:list)->float:
            return float()
    trainer = TestTrainer(10)
    cross = CrossValidationMaker(2, [1,2], trainer)

    cross.cross_validation()
    print('success')

if __name__ == "__main__":
    main()