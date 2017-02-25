#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np
import pandas as pd

import sys; sys.path.insert(0,'/Library/Frameworks/GDAL.framework/Versions/2.1/Python/2.7/site-packages')
from osgeo import ogr, osr
from simpledbf import Dbf5

class ShpRegistryExc(Exception):
    pass


class ShpRegistry:
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    srs = osr.SpatialReference()
    
    def __init__(self, shp_file_paths=None, shp_dir=None):
        self.shp_file_paths = shp_file_paths
        self.shp_dir = shp_dir
        
        if not any([self.shp_file_paths, self.shp_dir]):
            raise ShpRegistryExc('Need shp_file_paths, shp_dir')
        
        if self.shp_file_paths is not None:
            if type(self.shp_file_paths) != list:
                self.shp_file_paths = [self.shp_file_paths]
        else:
            self.shp_file_paths = []
        
        if self.shp_dir is None:
            self.shp_dir = []
        
        self.shp_file_paths.extend(self.__get_shp_file_paths())
        self.shp_file_paths = list(set(self.shp_file_paths))
            
    @staticmethod
    def __get_dataframe_from_shp(self, shp_file_path):
        shp_file = self.driver.Open(shp_file_path)
        dbf_filename = shp_file_path[:-4] + '.dbf'
        try:
            dbf_file = Dbf5(dbf_filename, codec='utf-8')
            df = dbf_file.to_dataframe()
        except UnicodeDecodeError:
            dbf_file = Dbf5(dbf_filename, codec='cp1251')
            df = dbf_file.to_dataframe()
        except Exception as e:
            raise ShpRegistryExc('Problem with data in file:', dbf_filename, e)
        
        try:
            daLayer = shp_file.GetLayer()
            wkt_geom = [feature.geometry().ExportToWkt() for feature in daLayer]
            spatialRef = daLayer.GetSpatialRef()
            df['_wkt_'] = wkt_geom
            df['_geom_type_'] = df['_wkt_'].str.findall(r'[A-z]+')
            df['_geom_type_'] = df['_geom_type_'].apply(lambda x: x[0])
            df['_prj_'] = spatialRef.ExportToWkt()
        except AttributeError as e:
            raise ShpRegistryExc('Problem with geometry in file:', shp_file_path, e)
        
        return df
    
    def __get_shp_file_paths(self):
        return [os.path.join(self.shp_dir, sf) for sf in os.listdir(self.shp_dir) if sf.split('.')[-1]=='shp']
    
    @staticmethod
    def __get_df_with_transform_prj(df, target_prj):
        
        def transform_geom_prj(geom, transform):
            geom.Transform(transform)
            return geom
        
        target = osr.SpatialReference()
        df = df[~pd.isnull(df['_prj_'])]
        if type(target_prj) == int:
            target.ImportFromEPSG(target_prj)
        elif type(target_prj) == str:
            target.ImportFromWkt(target_prj)
        elif not target_prj:
            target.ImportFromEPSG(4326)
        else:
            raise ShpRegistryExc('Incorrect projection:', target_prj)
            
        for prj_type in df['_prj_'].unique():
            source_prj = osr.SpatialReference()
            source_prj.ImportFromWkt(prj_type)
            transform = osr.CoordinateTransformation(source_prj, target)
            df.loc[df['_prj_']==prj_type, '_geometry_'] = df.loc[df['_prj_']==prj_type, '_wkt_'].apply(lambda wkt: ogr.CreateGeometryFromWkt(wkt))
            df.loc[df['_prj_']==prj_type, '_geometry_'] = df.loc[df['_prj_']==prj_type, '_geometry_'].apply(lambda geom: transform_geom_prj(geom, transform))
            df.loc[df['_prj_']==prj_type, '_wkt_'] = df.loc[df['_prj_']==prj_type, '_geometry_'].apply(lambda geom: geom.ExportToWkt())
        return df
        
    def concat_df_shp(self, transform_prj=False, target_prj=4326):
        conc_df = pd.DataFrame()
        for sfp in self.shp_file_paths:
            try:
                df_shp = self.__get_dataframe_from_shp(self, sfp)
            except ShpRegistryExc as e:
                print(e)
                continue
            if conc_df.empty:
                conc_df = df_shp
            else:
                conc_df = conc_df.append(df_shp, ignore_index=True)
            
        if transform_prj:

            conc_df = self.__get_df_with_transform_prj(conc_df, target_prj)
        
        return conc_df
    
    @staticmethod
    def concat_df_column_ids(*args):
        def concat(*args):
            strs = []
            for arg in args:
                if pd.isnull(arg):
                    continue
                else:
                    try:
                        arg = str(int(arg))
                    except ValueError:
                        arg = str(arg)
                strs.append(arg)
            return ','.join(strs) if strs else np.nan
        np_concat = np.vectorize(concat)
        return np_concat(*args)
    
    @staticmethod
    def get_point_xy(df):
        # point_xy = df.loc[df['_geom_type_'] == 'POINT', '_wkt_'].str.findall(r'[0-9.]+').as_matrix()
        # df.loc[df['_geom_type_'] == 'POINT', 'point_xy'] = point_xy
        return df