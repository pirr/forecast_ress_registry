#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:11:16 2017

@author: aleksejsmaga
"""

import os
import pandas as pd
import numpy as np
from shpregistry import ShpRegistry
from registry import RegistryFormatter, REGISTRY_COLUMNS
from merge import MergedXlsShpDf
from computing import GroupComputing
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
import time


def get_predict_scores(y_true, y_pred):
    predict_scores = {
                     # 'acc': [],
                      # 'for': [],
                      # 'npv': [],
                      # 'ppv': [],
                      # 'fdr': [],
                      # 'sens': [],
                      # 'spec': [],
                      'err': [],
                      # 'acc-err': [],
                      'fp': [],
                      'fn': [],
                      'tp': []
                      }

    def compute_predict_score(true_n_rows, pred_n_rows, predict_scores=predict_scores):
        tp = len(set(pred_n_rows) & set(true_n_rows))
        fp = len(set(pred_n_rows) - set(true_n_rows))
        fn = len(set(true_n_rows) - set(pred_n_rows))
        # tn = len(true_n_rows) - tp - fp - fn
        err = float(fp + fn)
        predict_scores['err'].append(err)
        predict_scores['fp'].append(fp)
        predict_scores['fn'].append(fn)
        predict_scores['tp'].append(tp)
        # predict_scores['acc'].append(acc)
        # predict_scores['acc-err'].append(tp - err)
        # predict_scores['acc'].append(float(tp + tn) / float(len(y_true))),
        # predict_scores['for'].append(float(fn) / float(fn + tn)),
        # predict_scores['npv'].append(float(tn) / float(fn + tn)),
        # predict_scores['ppv'].append(float(tp) / float(tp + fp)),
        # predict_scores['fdr'].append(float(fp) / float(tp + fp)),
        # predict_scores['sens'].append(float(tp) / float(tp + fn)),
        # predict_scores['spec'].append(float(tn) / float(fp + tn))

    unique_n_objs_pred = []
    unique_n_objs_true = []
    # group_count = float(len(np.unique(y_true)))
    row_count = len(y_true)

    for i, _ in enumerate(y_true):
        if (y_pred[i] in unique_n_objs_pred) or (y_true[i] in unique_n_objs_true):
            continue

        pred_n_rows = np.argwhere(y_pred == y_pred[i]).ravel()
        true_n_rows = np.argwhere(y_true == y_true[i]).ravel()
        compute_predict_score(true_n_rows, pred_n_rows)
        unique_n_objs_pred.append(y_pred[i])
        unique_n_objs_true.append(y_true[i])

    return predict_scores

shp_dir = ur'data//shps//'
shp_registry = ShpRegistry(shp_dir=shp_dir)
df_shp_registry = shp_registry.concat_df_shp(transform_prj=True)

df_shp_registry['point_xy'] = np.nan
df_shp_registry['point_xy'] = df_shp_registry['point_xy'].astype(object)
df_shp_registry = shp_registry.get_point_xy(df_shp_registry)
df_shp_registry['geom_id'] = shp_registry.concat_df_column_ids(
                                                               df_shp_registry['N_TKOORD'],
                                                               df_shp_registry['NOM_PU'],
                                                               df_shp_registry['priv'],
                                                               df_shp_registry['ID_PRIV'],
                                                               df_shp_registry['regid'],
                                                               # df_shp_registry['nomstr'],
                                                               # df_shp_registry['cnigri_id'],
                                                               # df_shp_registry['N_reestr_s']
                                                               )

df_shp_registry = df_shp_registry[['point_xy', 'geom_id', '_prj_', '_geometry_', '_wkt_', '_geom_type_']]
# df_shp_registry['N_TKOORD'],
# df_shp_registry['nomstr']
# df_shp_registry['cnigri_id']
# df_shp_registry['N_reestr_s']

poly_to_point = True
subj = u'alt-kr'
reestr_file = u'reestr_ALK-3005.xls'
file_path = os.path.join(u'data', subj)
results_dir = os.path.join(file_path, 'group_results')

if not os.path.exists(file_path):
    raise Exception('No file {}'.format(file_path))

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

df = pd.read_excel(os.path.join(file_path, reestr_file), skiprows=1)
registry_fmt = RegistryFormatter(df, registry_cols_dict=REGISTRY_COLUMNS)
registry_fmt.format(grand_taxons=False)
xls_registry = registry_fmt.registry

xls_registry = xls_registry[xls_registry['actual'] == u'А']

merged_df = MergedXlsShpDf(xls_registry, df_shp_registry)

merged_df.paste_xy_for_none_shp_obj()
mdf = merged_df.df[:100]

# mdf.to_pickle('full_ak-registry.pk')

merge_df_norm_coord = mdf[mdf['coord_checked'] == 'ok']
merge_df_err_coord = mdf

# {'max_similar_coef': 0.4, 'name_ratio': 69.0, 'dist_penalty_coef': 190.0}
# {'max_similar_coef': 0.28, 'name_ratio': 86.0, 'dist_penalty_coef': 120.0}
# {'max_similar_coef': 0.43, 'name_ratio': 53.0, 'dist_penalty_coef': 110.0}
# {'max_similar_coef': 0.2, 'name_ratio': 90, 'dist_penalty_coef': 200}
# {'max_similar_coef': 0.39, 'name_ratio': 85.0, 'dist_penalty_coef': 130.0}

def foo(params):

    clf = GroupComputing(df=mdf, **params)
    clf.set_groups()
    m = clf.df[['N', 'N_obj']].sort_values('N')
    copy_m = m.copy()

    n_obj_dict = {}
    obj_new_num = 0
    for n_row, n_obj in m.as_matrix():
        if pd.isnull(n_obj):
            obj_num = obj_new_num
            obj_new_num += 1
        elif n_obj in n_obj_dict:
            obj_num = n_obj_dict[n_obj]
        else:
            obj_num = obj_new_num
            n_obj_dict[n_obj] = obj_new_num
            obj_new_num += 1
        copy_m.loc[copy_m['N'] == n_row, 'N_obj'] = obj_num


    y_pred = copy_m['N_obj'].as_matrix().ravel()
    y_true = pd.read_csv('data//y_kem.csv', header=None).as_matrix().ravel()

    predict_scores = get_predict_scores(y_true, y_pred)
    mean_scores = {k + '_mean': float(sum(v)) / float(len(v)) for k, v in predict_scores.items()}
    print 'params', params
    print 'mean score:', mean_scores['err_mean']

    return {'loss': mean_scores['err_mean'], 'status': STATUS_OK}


space4grouping = {
    'dist_penalty_coef': hp.quniform('dist_penalty_coef', 100., 600., 10.),
    'max_similar_coef': hp.quniform('max_similar_coef', 0.18, 0.5, 0.01),
    'name_ratio': hp.quniform('name_ratio', 70, 95, 1),
    'coeff_for_diff_doc_type': hp.quniform('coeff_for_diff_doc_type', 0.01, 0.2, 0.01),
}

best = {'max_similar_coef': 0.75,
        'name_ratio': 92.0,
        'dist_penalty_coef': 250.0,
        'coeff_for_diff_doc_type': 0.3}
# trials = Trials()
# best = fmin(foo, space4grouping, algo=tpe.suggest, max_evals=100, trials=trials)
#
# print 'best:'
# print best
#
best['poly_to_point'] = poly_to_point
group_comp = GroupComputing(mdf, **best)
group_comp.set_groups()

if poly_to_point:
    point_pref = '-points-'
else:
    point_pref = '-'
csv_file_path = os.path.join(results_dir, subj + point_pref + time.strftime("%m%d-%H_%M_%S") + '.csv')

group_comp.df[['N', 'N_objectX', 'doc_type',
               'adm_distr', 'list_200', 'name_obj',
               'analysis_name', 'isnedra_pi', 'norm_pi',
               'lon', 'lat', '_geom_type_']].to_csv(csv_file_path, encoding='cp1251', sep='\t')


# writer = pd.ExcelWriter('data//3105-1-alk-full.xls')
# group_comp.df.to_excel(writer, 'group')
# writer.save()
# writer.close()
# m = group_comp.df[['N', 'N_obj']]
# m.to_csv('data/2305-1-alk.csv', sep=';')
# copy_m = m.copy()

# n_obj_dict = {}
# obj_new_num = 0
# for n_row, n_obj in m.as_matrix():
#     if pd.isnull(n_obj):
#         obj_num = obj_new_num
#         obj_new_num += 1
#     elif n_obj in n_obj_dict:
#         obj_num = n_obj_dict[n_obj]
#     else:
#         obj_num = obj_new_num
#         n_obj_dict[n_obj] = obj_new_num
#         obj_new_num += 1
#     copy_m.loc[copy_m['N'] == n_row, 'N_obj'] = obj_num
#
# y_pred = copy_m['N_obj'].as_matrix().ravel()
# y_true = pd.read_csv('data//y_kem.csv', header=None).as_matrix().ravel()
# predict_scores = get_predict_scores(y_true, y_pred)
# mean_score = {k + '_mean': float(sum(v)) / float(len(v)) for k, v in predict_scores.items()}
# err = sum(predict_scores['err'])
# tp = sum(predict_scores['tp'])
# print err, tp, (tp / (err + tp))
# writer = pd.ExcelWriter('data//group_irk.xls')
# group_comp.df.to_excel(writer, 'group')
# writer.save()
# writer.close()
# copy_m.to_csv('data//pred_x.csv', sep=';')

# print 'groups', group_comp_norm_coord.df
# group_comp_norm_coord.get_analysis_name_matrix

# group_comp_norm_coord.set_group_for_lower_dist

# a = group_comp_norm_coord.df['analysis_name']
# print group_comp_norm_coord
# group_comp_err_coord = GroupComputing(merge_df_err_coord, dist_group=group_comp_norm_coord.dist_group)
# group_comp_err_coord.set_group_for_lower_dist

# print(group_comp_err_coord.df)
# pass
# compute = GroupComputing(merged_df.df)
# PM = compute.get_compute_point_matrix()
# dist100_indxs = compute.get_lower_dist_true_indx(100)
# dist15000_indxs = compute.get_between_dist_true_indx(100, 15000)
