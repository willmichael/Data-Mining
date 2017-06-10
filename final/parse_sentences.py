import re
import pandas as pd

def import_data():
    print "Importing Data"
    trainLoc = './small_train.csv'

    dfTrain = pd.read_csv(trainLoc, header=None)
    dfTrain = dfTrain.values[:,:]

    return dfTrain

def cut_data(data):
    data = data[:,3:6]
    # print data
    return data

def remove_nonalphanum(data):
    for i,d in enumerate(data):
        for j,e in enumerate(d):
            if isinstance(e, basestring):
                pattern = re.compile('([^\s\w]|_)+')
                strippedList = pattern.sub('', e)
                data[i][j] = strippedList
    return data

def main():
    file_name = 'train_parsed_tmp.csv'
    data = import_data()
    data = cut_data(data)
    data = remove_nonalphanum(data)

    data = pd.DataFrame(data)
    data.to_csv(file_name, sep=',', header=None)


if __name__ == '__main__':
    main()
