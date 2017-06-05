#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import re


class MergedXlsShpDf:


    def __init__(self, xls_df, shp_df):
        self.df = self.__get_merged_xls_with_shp_data(xls_df, shp_df)
        # self.paste_geom_point_for_none_shp_obj()

    @staticmethod
    def __get_merged_xls_with_shp_data(xls_df, shp_df):
        merged = pd.merge(xls_df, shp_df, left_on='N_poly_table', right_on='geom_id', how='left')
        # merged = merged[~pd.isnull(merged['actual'])]
        # return merged[(~merged.duplicated('N', )) | (merged['N'].isnull())]
        return merged

    def paste_xy_for_none_shp_obj(self):
        self.df.loc[self.df['_geom_type_'] == 'POINT', 'lon'] = \
            self.df.loc[self.df['_geom_type_'] == 'POINT', '_wkt_'].apply(lambda wkt: re.findall(r'[0-9.]+', wkt)[0])
        self.df.loc[self.df['_geom_type_'] == 'POINT', 'lat'] = \
            self.df.loc[self.df['_geom_type_'] == 'POINT', '_wkt_'].apply(lambda wkt: re.findall(r'[0-9.]+', wkt)[1])
        self.df.loc[
                (~pd.isnull(self.df['lon']) & pd.isnull(self.df['_geom_type_'])), '_geom_type_'] = 'POINT'