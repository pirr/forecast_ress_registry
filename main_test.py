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

shp_dir = ur'/Users/aleksejsmaga/repository/notebook/forecast_ress_gis'
shp_registry = ShpRegistry(shp_dir=shp_dir)
df_shp_registry = shp_registry.concat_df_shp(transform_prj=True)

df_shp_registry['point_xy'] = np.nan
df_shp_registry['point_xy'] = df_shp_registry['point_xy'].astype(object)
df_shp_registry = shp_registry.get_point_xy(df_shp_registry)
df_shp_registry['geom_id'] = shp_registry.concat_df_column_ids(
                                                               df_shp_registry['nomstr'],
                                                               df_shp_registry['cnigri_id'],
                                                               df_shp_registry['N_reestr_s']
                                                               )

# df_shp_registry['N_TKOORD'],
# df_shp_registry['nomstr']
# df_shp_registry['N_reestr_s']

df = pd.read_excel(u'//Users//aleksejsmaga//temp//reestr.xls')
registry_fmt = RegistryFormatter(df, registry_cols_dict=REGISTRY_COLUMNS)
registry_fmt.format()
xls_registry = registry_fmt.registry

merged_df = MergedXlsShpDf(xls_registry, df_shp_registry)
merged_df.paste_xy_for_none_shp_obj()
mdf = merged_df.df
merge_df_norm_coord = mdf[mdf['coord_checked'] == 'ok']
merge_df_err_coord = mdf

group_comp = GroupComputing(mdf)
groups = group_comp.get_similar_groups()

print('DONE!')