#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:11:16 2017

@author: aleksejsmaga
"""

import pandas as pd
import numpy as np
from shpregistry import ShpRegistry
from registry import RegistryFormatter, REGISTRY_COLUMNS
from merge import MergedXlsShpDf
from computing import GroupComputing

shp_dir = ur'/Users/aleksejsmaga/repository/notebook/cnigri_gis'
shp_registry = ShpRegistry(shp_dir=shp_dir)
df_shp_registry = shp_registry.concat_df_shp(transform_prj=True)

df_shp_registry['point_xy'] = np.nan
df_shp_registry['point_xy'] = df_shp_registry['point_xy'].astype(object)
df_shp_registry = shp_registry.get_point_xy(df_shp_registry)
df_shp_registry['geom_id'] = shp_registry.concat_df_column_ids(df_shp_registry['cnigri_id'],
                                                               df_shp_registry['nomstr_1'],
                                                               df_shp_registry['N_reestr_s'])

df = pd.read_excel(u'//Users//aleksejsmaga//repository//notebook//reestr_4субъекта_2.xls')
registry_fmt = RegistryFormatter(df, REGISTRY_COLUMNS)
registry_fmt.format()
xls_registry = registry_fmt.registry

merged_df = MergedXlsShpDf(xls_registry, df_shp_registry)
merged_df.paste_xy_for_none_shp_obj()
mdf = merged_df.df
merge_df_norm_coord = mdf[mdf['coord_checked'] == 'ok']
merge_df_err_coord = mdf

group_comp = GroupComputing(mdf)
# print group_comp_norm_coord.analysis_name_matrix
# print group_comp_norm_coord.df['_geom_type_']
# print group_comp_norm_coord.compare_polygons_matrix
# print group_comp_norm_coord.analysis_name_matrix

group_comp.set_groups(err_coord=True)
writer = pd.ExcelWriter('group_3_2.xls')
group_comp.df.to_excel(writer, 'group')
writer.save()
writer.close()
# group_comp.df.to('groups.csv', encoding='cp1251')

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
