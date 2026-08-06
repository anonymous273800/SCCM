from tabulate import tabulate


def print_list(list_, kpi):
    for i in range(len(list_)):
        print("mini-batch-", i + 1, kpi, ":  ", "{:.5f}".format(list_[i]))


def print_list_tabulate(list_):
    headers = [""] * len(list_)  # Empty string for all headers
    data = [["m-b-" + str(i + 1)] + ["{:.5f}".format(list_[i])] for i in range(len(list_))]
    transposed_data = list(map(list, zip(*data)))
    print(tabulate(transposed_data, headers=headers, tablefmt="plain"))
