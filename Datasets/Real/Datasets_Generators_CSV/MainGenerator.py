import pandas as pd

from Datasets.SyntheticDS.Abrupt import DS01, DS02, DS03, DS04
from Datasets.SyntheticDS.Gradual import DS09, DS10, DS11, DS12
from Datasets.SyntheticDS.Incremental import DS05, DS06, DS07, DS08
from Utils import Util


def generate_excel_from_DS(X, y, directory, file_name):
    n_samples, n_features = X.shape

    path_to_save_data_set = Util.get_path_to_save_generated_dataset_file(directory)
    Util.create_directory(path_to_save_data_set)
    # Create a DataFrame using X and y
    df = pd.DataFrame(data=X, columns=[f"feature_{i}" for i in range(n_features)])
    df['target'] = y

    # Save DataFrame to a CSV file
    df.to_csv(path_to_save_data_set + '\\' + file_name + '.csv', index=False)


if __name__ == '__main__':
    print('Abrupt Datasets')
    X, y = DS01.get_DS01()
    generate_excel_from_DS(X, y, 'Abrupt', '001_DS1')
    X, y = DS02.get_DS02()
    generate_excel_from_DS(X, y, 'Abrupt', '002_DS2')
    X, y = DS03.get_DS03()
    generate_excel_from_DS(X, y, 'Abrupt', '003_DS3')
    X, y = DS04.get_DS04()
    generate_excel_from_DS(X, y, 'Abrupt', '004_DS4')

    print("Incremental Datasets.")
    X, y = DS05.get_DS05()
    generate_excel_from_DS(X, y, 'Incremental', '005_DS5')
    X, y = DS06.get_DS06()
    generate_excel_from_DS(X, y, 'Incremental', '006_DS6')
    X, y = DS07.get_DS07()
    generate_excel_from_DS(X, y, 'Incremental', '007_DS7')
    X, y = DS08.get_DS08()
    generate_excel_from_DS(X, y, 'Incremental', '008_DS8')

    print("Gradual Datasets.")
    X, y = DS09.get_DS09()
    generate_excel_from_DS(X, y, 'Gradual', '009_DS9')
    X, y = DS10.get_DS10()
    generate_excel_from_DS(X, y, 'Gradual', '010_DS10')
    X, y = DS11.get_DS11()
    generate_excel_from_DS(X, y, 'Gradual', '011_DS11')
    X, y = DS12.get_DS12()
    generate_excel_from_DS(X, y, 'Gradual', '012_DS12')
