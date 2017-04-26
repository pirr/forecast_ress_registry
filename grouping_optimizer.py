from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


train_reestr = pd.read_excel('data//y_train.xls')
result_y = train_reestr.copy()

m = train_reestr.as_matrix()
n_obj_dict = {}
obj_new_num = 0
for n_row, n_obj in train_reestr.as_matrix():
    if n_obj in n_obj_dict:
        obj_num = n_obj_dict[n_obj]
    else:
        obj_num = obj_new_num
        n_obj_dict[n_obj] = obj_new_num
        obj_new_num += 1
    result_y.loc[result_y['N'] == n_row, 'N_obj'] = obj_num

result_y.to_csv('data//y_train.csv', sep=';')


