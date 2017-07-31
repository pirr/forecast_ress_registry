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
import sys


poly_to_point = True
subj = u'hak'
reestr_file = u'mz-hak.xlsx'
file_path = os.path.join(u'data', subj)
results_dir = os.path.join(file_path, 'group_results')

group_small_ojs = False
group_big_objs = True

best = {'max_similar_coef': 0.9,
        'name_ratio': 92.0,
        'dist_penalty_coef': 2500.0,
        'coeff_for_diff_doc_type': 0.2,
        'max_dist': 20000,
        'processes': 3}
best['poly_to_point'] = poly_to_point



big_obj_types = [u'мз', u'пп', u'рз', u'рп', u'рр', u'ру']

if __name__ == '__main__':
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

  df_shp_registry = df_shp_registry[
      ['point_xy', 'geom_id', '_prj_', '_geometry_', '_wkt_', '_geom_type_']]

  if not os.path.exists(file_path):
    msg = u'File {} not exists'.format(file_path)
    raise Exception(msg)

  if not os.path.exists(results_dir):
    os.makedirs(results_dir)

  df = pd.read_excel(os.path.join(file_path, reestr_file), skiprows=1)
  registry_fmt = RegistryFormatter(df, registry_cols_dict=REGISTRY_COLUMNS)

  registry_fmt.format(grand_taxons=True)

  xls_registry = registry_fmt.registry

  res_df = pd.DataFrame()

  if group_big_objs:
    print('!!!BIG OBJS GROUPING!!!"')
    for select in big_obj_types:
      xls_registry_ = xls_registry[(xls_registry[
          'actual'] == u'А') & (xls_registry['geol_type_obj'].str.lower() == select)]

      merged_df = MergedXlsShpDf(xls_registry_, df_shp_registry)

      merged_df.paste_xy_for_none_shp_obj()
      mdf = merged_df.df

      if mdf.empty:
        print(u'{} empty or not coordinates'.format(select))
        continue

      group_comp = GroupComputing(mdf, **best)
      group_comp.set_groups()

      df = group_comp.df[['N', 'complex', 'old_N_objectX', 'N_objectX', 'doc_type',
                          'adm_distr', 'list_200', 'name_obj',
                          'analysis_name', 'len_analysis_name', 'isnedra_pi', 'norm_pi',
                          'lon', 'lat', '_geom_type_']]
      # df.to_csv(csv_file_path, encoding='cp1251', sep='\t')

      res_df = pd.concat([res_df, df])

  if group_small_ojs:
    print('!!!SMALL OBJS GROUPING!!!"')

    xls_registry_ = xls_registry[(xls_registry['actual'] == u'А') & (~xls_registry['geol_type_obj'].str.lower().isin(big_obj_types))]
    merged_df = MergedXlsShpDf(xls_registry_, df_shp_registry)

    merged_df.paste_xy_for_none_shp_obj()
    mdf = merged_df.df

    if mdf.empty:
      print(u'{} empty or not coordinates'.format(select))
      sys.exit()

    group_comp = GroupComputing(mdf, **best)
    group_comp.set_groups()

    df = group_comp.df[['N', 'complex', 'old_N_objectX', 'N_objectX', 'doc_type',
                            'adm_distr', 'list_200', 'name_obj',
                            'analysis_name', 'len_analysis_name', 'isnedra_pi', 'norm_pi',
                            'lon', 'lat', '_geom_type_']]
    res_df = pd.concat([res_df, df])

  if poly_to_point:
    point_pref = '-points-'
  else:
    point_pref = '-'

  if group_big_objs:
    kt_pref = 'kt-'
  else:
    kt_pref = ''
  csv_file_path = os.path.join(
      results_dir, kt_pref + subj + point_pref + time.strftime("%m%d-%H_%M_%S") + '.csv')

  res_df.to_csv(csv_file_path, encoding='cp1251', sep='\t')
  # writer = pd.ExcelWriter('data//3105-1-alk-full.xls')
  # group_comp.df.to_excel(writer, 'group')
  # writer.save()
  # writer.close()
