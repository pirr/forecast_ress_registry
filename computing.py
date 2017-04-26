#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:19:57 2017

@author: aleksejsmaga
"""

import string
from datetime import datetime
from itertools import combinations, product, izip
from functools import partial
import re
import numpy as np
import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz
from sklearn.neighbors import DistanceMetric
from multiprocessing import Pool

import sys; sys.path.insert(0, '/Library/Frameworks/GDAL.framework/Versions/2.1/Python/2.7/site-packages')
from osgeo import ogr


class GroupComputing:
    exclude = re.compile('[%s]' % string.punctuation)
    name_pattern = pd.read_csv(
        u'dict//pattern_for_replace.csv', sep=';', encoding='cp1251').fillna('')

    def __init__(self, df, **kwargs):
        self.df = df
        self.processes = kwargs.get('processes', 4)
        self.err_coord = kwargs.get('err_coord', False)
        self.group_num = self.get_last_group_num
        self.lon_field = kwargs.get('lon_field', 'lon')
        self.lat_field = kwargs.get('lat_field', 'lat')
        self.min_dist = kwargs.get('min_dist', 0.)
        self.max_dist = kwargs.get('max_dist', 5000.)
        self.err_dist = kwargs.get('err_dist', 50000.)
        self.geometry_field = kwargs.get('geometry', '_geometry_')
        self.df[self.lon_field] = self.df[self.lon_field].astype(float)
        self.df[self.lat_field] = self.df[self.lat_field].astype(float)
        self.df['analysis_name'] = self.df['name_obj'].str.lower()
        self.__clear_analysis_names()
        self.point_indxs = self.__get_point_indxs
        self.computed_point_matrix = self.get_compute_point_matrix
        self.buffer_dist = kwargs.get('buffer_dist', 1)
        self.name_ratio = kwargs.get('name_ratio', 90)
        self.dist_penalty_coef = float(kwargs.get('dist_penalty_coef', 100))
        self.max_similar_coef = float(kwargs.get('max_similar_coef', 0.35))
        self.coeff_for_diff_doc_type = float(kwargs.get('coeff_for_diff_doc_type', 1.3))


        # self.df['group_pi'] = self.df['isnedra_pi']
        # self.name_score_matrix = self.get_name_score_matrix()

    def __clear_analysis_names(self):
        for p in self.name_pattern.as_matrix():
            self.df['analysis_name'] = self.df[
                'analysis_name'].str.replace(ur'{}'.format(p[0]), p[1], flags=re.UNICODE)
            self.df['analysis_name'] = self.df['analysis_name'].str.strip()

    @property
    def __get_point_radian_matrix(self):
        point_matrix = self.df.loc[self.df['_geom_type_'] == 'POINT', [
            self.lat_field, self.lon_field]]
        return point_matrix / 57.29578

    @property
    def __get_point_indxs(self):
        return self.df[self.df['_geom_type_'] == 'POINT'].index.values.tolist()

    @property
    def get_compute_point_matrix(self):
        point_radian_matrix = self.__get_point_radian_matrix
        dist = DistanceMetric.get_metric(metric='haversine')
        D = dist.pairwise(point_radian_matrix)
        for i in xrange(len(D)):
            D[i, :i + 1] = np.nan
        return D * 6372795.

    @property
    def get_lower_dist_groups(self):
        computed_lower_dist = np.where(self.computed_point_matrix <= self.min_dist,
                                       self.computed_point_matrix, np.nan)
        G = self.get_groups_graph(np.argwhere(~np.isnan(computed_lower_dist)))
        return self.get_reindexed_group_from_graph(G, self.point_indxs)

    @property
    def get_between_dist_groups(self):
        computed_between_dist = np.where(
            (self.computed_point_matrix > self.min_dist) & (
                self.computed_point_matrix <= self.max_dist),
            self.computed_point_matrix, np.nan)
        G = self.get_groups_graph(np.argwhere(~np.isnan(computed_between_dist)))
        return self.get_reindexed_group_from_graph(G, self.point_indxs)

    @property
    def get_between_err_coord_dist_groups(self):
        computed_between_dist = np.where(
            (self.computed_point_matrix > self.max_dist) & (
                self.computed_point_matrix <= self.err_dist),
            self.computed_point_matrix, np.nan)
        not_err_coords_indxs = self.df[(self.df['coord_checked'] == 'ok') & (self.df['_geom_type_'] == 'POINT')].index.values.tolist()
        not_err_coords_order_indxs = [self.point_indxs.index(i) for i in not_err_coords_indxs]
        computed_between_dist = np.delete(computed_between_dist, not_err_coords_order_indxs, axis=0)
        G = self.get_groups_graph(np.argwhere(~np.isnan(computed_between_dist)))
        return self.get_reindexed_group_from_graph(G, self.point_indxs)

    @property
    def get_analysis_name_groups(self):
        ratio = self.name_ratio
        name_rows = izip(self.df.index.values.tolist(), self.df['analysis_name'])
        combo = combinations(name_rows, r=2)
        p = Pool(self.processes)
        groups = p.map(partial(_fuzz_part_ratio, ratio), combo)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        return nx.connected_components(G)

    @property
    def get_last_group_num(self):
        num = self.df['N_obj'].max(skipna=True)
        if pd.isnull(num):
            return 1
        return num + 1

    @property
    def get_group_pi_groups(self):
        group_pi_rows = izip(self.df.index.values.tolist(), self.df['group_pi'].values.tolist())
        combo = combinations(group_pi_rows, r=2)
        p = Pool(self.processes)
        groups = p.map(_compare_group_pi, combo)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        return nx.connected_components(G)

    @property
    def get_attribute_groups(self):
        attrs_rows = izip(self.df.index.values.tolist(), self.df[['analysis_name', 'group_pi']].values.tolist())
        combo = combinations(attrs_rows, r=2)
        p = Pool(self.processes)
        name_ratio = self.name_ratio
        groups = p.map(partial(_get_attribute_groups, name_ratio), combo)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        return list(nx.connected_components(G))

    @property
    def get_polygon_by_wkt_groups(self):
        polygons_df = self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']),
                                  ['_geometry_', '_wkt_', 'analysis_name']]
        indxs = polygons_df.index.values.tolist()
        polygons_df = izip(indxs, polygons_df.to_dict(orient='records'))
        polygons_prod = combinations(polygons_df, r=2)
        p = Pool(self.processes)
        groups = p.map(_compare_polygons_wkt, polygons_prod)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        return nx.connected_components(G)

    @property
    def get_polygon_iters_witin_groups(self):
        polygons_df = self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']),
                                  ['_geometry_', '_wkt_', 'analysis_name']]
        indxs = polygons_df.index.values.tolist()
        polygons_df = izip(indxs, polygons_df.to_dict(orient='records'))
        polygons_prod = combinations(polygons_df, r=2)
        p = Pool(self.processes)
        groups = p.map(_compare_polygons_inter_within, polygons_prod)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        return nx.connected_components(G)

    @property
    def compute_polygon_point_groups(self):
        polygons = self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']), '_geometry_']
        points = self.df.loc[self.df['_geom_type_'] == 'POINT', ['lon', 'lat', 'coord_checked']]
        groups = []
        for point_row in points.iterrows():
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(point_row[1]['lon'], point_row[1]['lat'])

            for poly in polygons.iteritems():
                if point.Within(poly[1]):
                    # print 'point within poly', (point_row[0], poly[0])
                    groups.append([point_row[0], poly[0]])
                # if point_row[1]['coord_checked'] == 'err':
                #     buffer = point.Buffer(self.buffer_dist)
                #     if any([
                #         buffer.Intersection(poly[1]),
                #         buffer.Within(poly[1]),
                #         poly[1].Within(buffer)
                #     ]):
                #         groups.append([point_row[0], poly[0]])
                # else:
                #     if point.Within(poly[1]):
                #         # print 'point within poly', (point_row[0], poly[0])
                #         groups.append([point_row[0], poly[0]])
        G = self.get_groups_graph(np.array(groups))
        return nx.connected_components(G)

    @staticmethod
    def get_groups_graph(edges):
        G = nx.Graph()
        if edges.size:
            nodes = np.unique(edges)
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        return G

    @staticmethod
    def get_reindexed_group_from_graph(graph, df_index):
        groups = list(nx.connected_components(graph))
        reindexed_group = []
        for group in groups:
            reindexed_group.append(set([df_index[n] for n in group]))
        return reindexed_group

    @staticmethod
    def get_merged_groups(*groups):
        merge_groups = []
        for gr_1 in groups[0]:
            for gr_2 in groups[1]:
                merge_gr = gr_1 & gr_2
                if len(merge_gr) > 1:
                    merge_groups.append(merge_gr)
        return merge_groups

    @staticmethod
    def get_merged_lists(*lists):
        merged_list = []
        for l in lists:
            merged_list.extend(l)
        return merged_list

    def get_full_groups(self, err_coord=False):
        print 'creating attribute groups'
        attr_groups = self.get_attribute_groups
        print 'creating similar point groups'
        similar_point_groups = list(self.get_similar_point_groups())
        # print 'creating between distance groups'
        # between_dist_groups = self.get_merged_groups(self.get_between_dist_groups, attr_groups)
        # between_err_coord_dist_groups = []
        # if err_coord:
        #     print 'creating err coord between distance groups'
        #     between_err_coord_dist_groups = self.get_merged_groups(self.get_between_err_coord_dist_groups, attr_groups)
        print 'creating polygon groups'
        polygon_groups = self.get_merged_groups(self.get_polygon_iters_witin_groups, attr_groups)
        print 'creating lower distance groups'
        lower_dist_groups = self.get_lower_dist_groups
        print 'creating polygon by wkt groups'
        polygon_by_wkt_groups = list(self.get_polygon_by_wkt_groups)
        print 'creating points within polygons groups'
        polygon_point_groups = self.get_merged_groups(self.compute_polygon_point_groups, attr_groups)
        return self.get_merged_lists(
                                     # between_dist_groups,
                                     # between_err_coord_dist_groups,
                                     similar_point_groups,
                                     polygon_groups,
                                     lower_dist_groups,
                                     polygon_by_wkt_groups,
                                     polygon_point_groups
                                     )

    def set_groups(self, err_coord=False):
        start = datetime.now()
        print 'start grouping'
        groups = self.get_full_groups(err_coord=err_coord)
        for group in groups:
            group = list(group)
            group_num = self.df.loc[group, 'N_obj'].min(skipna=True)
            if pd.isnull(group_num):
                group_num = self.group_num
                self.group_num += 1
            else:
                n_objs = self.df.loc[group, 'N_obj'].dropna().unique()
                self.df.loc[self.df['N_obj'].isin(n_objs), 'N_obj'] = group_num
            self.df.loc[group, 'N_obj'] = group_num
        print 'end grouping', datetime.now() - start

    def get_name_similar_ratio(self, names_list):
        names_combo = combinations(names_list, r=2)
        p = Pool(self.processes)
        names_similar_ratio = np.array(p.map(_fuzz_partial_ratio, names_combo))
        p.close()
        p.join()
        return names_similar_ratio

    def get_group_pi_equal(self, group_pi_list):
        group_pi_combo = combinations(group_pi_list, r=2)
        p = Pool(self.processes)
        return np.array(p.map(_group_pi_equal, group_pi_combo))

    def get_doc_type_ratio(self, doc_types_list, coeff_for_diff=1.3):
        doc_types_combos = combinations(doc_types_list, r=2)
        doc_types_ratio = [coeff_for_diff if len(set(doc_types)) == 2 else 0. for doc_types in doc_types_combos]
        return np.array(doc_types_ratio)

    def get_similar_point_groups(self):

        df_points = self.df[self.df['_geom_type_'] == 'POINT']

        name_similar_ratio = self.get_name_similar_ratio(df_points['analysis_name'].values.tolist())
        name_similar_ratio = name_similar_ratio.astype(float)
        dist_list = self.computed_point_matrix.ravel()
        dist_list = dist_list[~np.isnan(dist_list)]

        # _D = (dist_list + (self.dist_penalty_coef * (self.name_ratio-name_similar_ratio))) / self.max_dist
        # _R = (100. - name_similar_ratio) / 100.
        # similar_coeff = _D * _R
        doc_types_ratio = self.get_doc_type_ratio(df_points['doc_type'].values.tolist(), self.coeff_for_diff_doc_type)
        similar_coeff = ((dist_list + (self.dist_penalty_coef * (self.name_ratio-name_similar_ratio))) - doc_types_ratio) / self.max_dist

        group_pi_equal = self.get_group_pi_equal(df_points['group_pi'].values.tolist())
        indxs_combo = np.array(list(combinations(self.point_indxs, r=2)))
        similar_couple = indxs_combo[(similar_coeff <= self.max_similar_coef) & (group_pi_equal==True)]
        G = self.get_groups_graph(similar_couple)

        return nx.connected_components(G)


def _fuzz_part_ratio(ratio, names):
    name_1 = names[0][1]
    name_2 = names[1][1]
    if fuzz.partial_ratio(name_1, name_2) >= ratio:
        return names[0][0], names[1][0]


def _compare_polygons_inter_within(polygons):
    poly_1 = polygons[0][1]
    poly_2 = polygons[1][1]
    if any([
        poly_1['_geometry_'].Intersection(poly_2['_geometry_']),
        poly_1['_geometry_'].Within(poly_2['_geometry_']),
        poly_2['_geometry_'].Within(poly_1['_geometry_'])
    ]):
        return polygons[0][0], polygons[1][0]


def _get_attribute_groups(name_ratio, attrs):
    name_1 = attrs[0][1][0]
    name_2 = attrs[1][1][0]
    gr_pi1 = attrs[0][1][1]
    gr_pi2 = attrs[1][1][1]
    if fuzz.partial_ratio(name_1, name_2) >= name_ratio and (gr_pi1==gr_pi2):
        return attrs[0][0], attrs[1][0]


def _compare_group_pi(groups):
    group_pi_1 = groups[0][1]
    group_pi_2 = groups[1][1]
    if group_pi_1 == group_pi_2:
        return groups[0][0], groups[1][0]


def _compare_polygons_wkt(polygons):
    poly_1 = polygons[0][1]
    poly_2 = polygons[1][1]
    if poly_1['_wkt_'] == poly_2['_wkt_']:
        return polygons[0][0], polygons[1][0]


def _fuzz_partial_ratio(names):
    ratio = fuzz.partial_ratio(*names)
    return ratio

def _group_pi_equal(group_pi_couple):
    return group_pi_couple[0] == group_pi_couple[1]
