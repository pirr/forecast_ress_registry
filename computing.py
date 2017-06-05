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
from collections import Counter
import pymorphy2


import sys; sys.path.insert(0, '/Library/Frameworks/GDAL.framework/Versions/2.1/Python/2.7/site-packages')
from osgeo import ogr


class GroupComputing:



    def __init__(self, df, **kwargs):
        self.df = df
        self.processes = kwargs.get('processes', 4)
        self.err_coord = kwargs.get('err_coord', False)
        self.lon_field = kwargs.get('lon_field', 'lon')
        self.lat_field = kwargs.get('lat_field', 'lat')
        self.min_dist = kwargs.get('min_dist', 0.)
        self.max_dist = kwargs.get('max_dist', 5000.)
        self.err_dist = kwargs.get('err_dist', 50000.)
        self.geometry_field = kwargs.get('geometry', '_geometry_')
        self.buffer_dist = kwargs.get('buffer_dist', 1)
        self.name_ratio = kwargs.get('name_ratio', 90)
        self.dist_penalty_coef = float(kwargs.get('dist_penalty_coef', 100))
        self.max_similar_coef = float(kwargs.get('max_similar_coef', 0.35))
        self.coeff_for_diff_doc_type = float(kwargs.get('coeff_for_diff_doc_type', 1.3))
        self.poly_to_point = kwargs.get('poly_to_point', False)
        if self.poly_to_point:
            print '!!!POLY TO POINT ON!!!'
            self.__set_poly_centroids()
        self.group_num = self.get_last_group_num
        self.df[self.lon_field] = self.df[self.lon_field].astype(float)
        self.df[self.lat_field] = self.df[self.lat_field].astype(float)
        self.point_indxs = self.__get_point_indxs
        self.computed_point_matrix = self.get_compute_point_matrix
        self.df['analysis_name'] = self.__get_analysis_name()
        self.__clear_analysis_names()
        # pd.DataFrame(self.get_word_duplicates(count=2)).to_csv('data/duplicates.csv', sep=';', encoding='cp1251')
        self.df['isnedra_pi'] = self.df['isnedra_pi'].str.lower().str.strip()
        self.df['norm_pi'] = self.df['norm_pi'].str.lower().str.strip()
        # self.name_score_matrix = self.get_name_score_matrix()

    def __get_analysis_name(self):
        def catch_err_str(s):
            try:
                s = s.lower()
            except AttributeError as e:
                s = unicode(str(s), 'utf-8')
            return s

        return self.df['name_obj'].apply(lambda name: catch_err_str(name))

    def __clear_analysis_names(self):
        name_pattern = pd.read_csv(
            u'dict//pattern_for_replace.csv', sep=';', encoding='cp1251').fillna('')
        morph = pymorphy2.MorphAnalyzer()

        def normalize_words(words_str):
            norm_word_list = [morph.parse(w)[0].normal_form for w in words_str.split(' ')]
            return ' '.join(norm_word_list)
        self.df['analysis_name'] = self.df['analysis_name'].apply(lambda name: normalize_words(name))
        for p in name_pattern.as_matrix():
            self.df['analysis_name'] = self.df[
                'analysis_name'].str.replace(ur'{}'.format(p[0]), p[1], flags=re.UNICODE)
            self.df['analysis_name'] = self.df['analysis_name'].str.strip()
        self.df['analysis_name'] = self.df[
                'analysis_name'].str.replace(r'[%s]' % string.punctuation, ' ', flags=re.UNICODE)
        self.df['analysis_name'] = self.df[
            'analysis_name'].str.replace(r'\s\s+', ' ', flags=re.UNICODE).str.strip()

    def __set_poly_centroids(self):
        self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']) & pd.isnull(self.df['lon']), 'lon'] = \
            self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']) & pd.isnull(self.df['lon']), '_geometry_'].apply(
                lambda geom: re.findall(r'[0-9.]+', geom.Centroid().ExportToWkt())[0])
        self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']) & pd.isnull(self.df['lat']), 'lat'] = \
            self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']) & pd.isnull(self.df['lat']), '_geometry_'].apply(
                lambda geom: re.findall(r'[0-9.]+', geom.Centroid().ExportToWkt())[1])

        self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']), '_geom_type_'] = 'POINT'

    def get_word_duplicates(self, count=3):
        words = []
        for w in self.df['analysis_name']:
            words.extend(w.split(' '))
        counter = Counter(words)
        duplicates = [w for w, c in counter.items() if c >= count]

        return duplicates

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
    def get_points_lower_dist_groups(self):
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
        num = self.df['N_objectX'].max(skipna=True)
        if pd.isnull(num):
            return 1
        return num + 1

    @property
    def get_attribute_groups(self):
        attrs_rows = izip(self.df.index.values.tolist(),
                          self.df[['analysis_name', 'isnedra_pi', 'norm_pi']].values.tolist())
        combo = combinations(attrs_rows, r=2)
        p = Pool(self.processes)
        name_ratio = self.name_ratio
        groups = p.map(partial(_get_attrs_ratio, name_ratio), combo)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        indxs_set = set()
        cliques = []
        for clique in list(nx.find_cliques(G))[::-1]:
            clique = set(clique) - indxs_set
            if len(clique) > 1:
                cliques.append(clique)
                indxs_set = indxs_set | clique
        #         self.df.loc[list(clique), 'N_objectX'] = i
        #         i += 1
        # writer = pd.ExcelWriter('data/analysis_name-pi-hrouping.xls')
        # self.df.to_excel(writer, 'group')
        # writer.save()
        # writer.close()
        # stop
        # for comp in nx.connected_components(G):
        #     avg_weight = []
        # G = self.get_groups_graph(np.array(groups))
        # groups_df = pd.DataFrame(groups)
        # groups_df.to_csv('data/name_ratio.csv')

        return cliques
        # return nx.connected_components(G)

    def get_graph_by_func(self, df, func, *args):
        indxs = df.index.values.tolist()
        indxs_attrs = izip(indxs, df.to_dict(orient='records'))
        combos_attrs = combinations(indxs_attrs, r=2)
        p = Pool(self.processes)
        groups = p.map(partial(func, *args), combos_attrs)
        p.close()
        p.join()
        groups = [g for g in groups if g is not None]
        G = self.get_groups_graph(np.array(groups))
        indxs_set = set()
        cliques = []
        for clique in list(nx.find_cliques(G))[::-1]:
            clique = set(clique) - indxs_set
            if len(clique) > 1:
                cliques.append(clique)
                indxs_set = indxs_set | clique
        return cliques

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
                    groups.append([point_row[0], poly[0]])
                elif point.Distance(poly[1]) <= 0.09:
                    groups.append([point_row[0], poly[0]])
        G = self.get_groups_graph(np.array(groups))
        return nx.connected_components(G)

    @property
    def compute_line_poin_groups(self):
        pass

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
                    if 1274 in merge_gr:
                        print gr_1, gr_2
                        break
        return merge_groups

    @staticmethod
    def get_merged_lists(*lists):
        merged_list = []
        for l in lists:
            merged_list.extend(l)
        return merged_list

    def get_full_groups(self):
        if not self.poly_to_point:
            print 'creating attribute groups'
            attr_groups = list(self.get_attribute_groups)
            print 'creating polygon groups'
            polygons_df = self.df.loc[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON']),
                                      ['_geometry_', '_wkt_', 'analysis_name', 'isnedra_pi', 'norm_pi']]
            polygon_inters_within_groups = list(self.get_graph_by_func(polygons_df, _compare_polygons_inter_within))
            polygon_inters_within_groups_attrs = self.get_merged_groups(polygon_inters_within_groups, attr_groups)
            print 'creating polygon by wkt groups'
            polygon_by_wkt_groups = self.get_graph_by_func(polygons_df, _compare_polygons_wkt)
            print 'creating points within polygons groups'
            polygon_point_groups = self.get_merged_groups(self.compute_polygon_point_groups, attr_groups)

        else:
            polygon_inters_within_groups_attrs = []
            polygon_by_wkt_groups = []
            polygon_point_groups = []

        print 'creating similar point groups'
        similar_point_groups = list(self.get_similar_point_groups())
        print 'creating points lower distance groups'
        lower_dist_groups = self.get_points_lower_dist_groups
        print 'process set grouping..'
        return self.get_merged_lists(
                                     similar_point_groups,
                                     polygon_inters_within_groups_attrs,
                                     lower_dist_groups,
                                     polygon_by_wkt_groups,
                                     polygon_point_groups
                                     )

    def set_groups(self):
        start = datetime.now()
        print 'start grouping'
        groups = self.get_full_groups()
        for group in groups:
            group = list(group)
            group_num = self.df.loc[group, 'N_objectX'].min(skipna=True)
            if pd.isnull(group_num):
                group_num = self.group_num
                self.group_num += 1
            else:
                n_objs = self.df.loc[group, 'N_objectX'].dropna().unique()
                self.df.loc[self.df['N_objectX'].isin(n_objs), 'N_objectX'] = group_num
            self.df.loc[group, 'N_objectX'] = group_num

        print 'end grouping', datetime.now() - start

    def get_name_similar_ratio(self, names_list):
        names_combo = combinations(names_list, r=2)
        p = Pool(self.processes)
        names_similar_ratio = np.array(p.map(_fuzz_similar_ratio, names_combo))
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
        doc_types_ratio = self.get_doc_type_ratio(df_points['doc_type'].values.tolist(), self.coeff_for_diff_doc_type)

        dist_penalty = self.dist_penalty_coef * (self.name_ratio - name_similar_ratio)
        similar_coeff = ((dist_list + dist_penalty) / self.max_dist ) - doc_types_ratio
        group_pi_equal = self.get_group_pi_equal(df_points[['isnedra_pi', 'norm_pi']].values.tolist())
        indxs_combo = np.array(list(combinations(self.point_indxs, r=2)))
        similar_couple = indxs_combo[(similar_coeff <= self.max_similar_coef) & (group_pi_equal==True)]
        G = self.get_groups_graph(similar_couple)
        return nx.connected_components(G)

    def get_similar_polygon_groups(self):
        df_polygons = self.df[self.df['_geom_type_'].isin(['POLYGON', 'MULTIPOLYGON'])]
        name_similar_ratio = self.get_name_similar_ratio(df_polygons['analysis_name'].values.tolist())
        name_similar_ratio = name_similar_ratio.astype(float)

        polygon_combs = combinations(df_polygons['_wkt_', '_geometry_'], r=2)


def _fuzz_part_ratio(ratio, names):
    name_1 = names[0][1]
    name_2 = names[1][1]
    if fuzz.partial_ratio(name_1, name_2) >= ratio:
        return names[0][0], names[1][0]


def _compare_polygons_inter_within_2(polygons):
    poly_1 = polygons[0][1]
    poly_2 = polygons[1][1]

    geometry_similar = 0

    if poly_1['_wkt_'] == poly_2['_wkt_']:
        geometry_similar = 1
    elif poly_1['_geometry_'].Within(poly_2['_geometry_']) or poly_2['_geometry_'].Within(poly_1['_geometry_']):
        geometry_similar = 0.7
    elif poly_1['_geometry_'].Intersection(poly_2['_geometry_']):
        geometry_similar = 0.5
    elif poly_1['_geometry_'].Distance(poly_2['_geometry_']) <= 0.01:
        geometry_similar = 0.3

    return geometry_similar


def _compare_polygons_inter_within(polygons):
    poly_1 = polygons[0][1]
    poly_2 = polygons[1][1]
    if any([
        poly_1['_geometry_'].Intersect(poly_2['_geometry_']),
        poly_1['_geometry_'].Within(poly_2['_geometry_']),
        poly_2['_geometry_'].Within(poly_1['_geometry_'])
    ]):
        return polygons[0][0], polygons[1][0]

def _compare_name_pi(name_ratio, objs):
    obj1 = objs[0][1]
    obj2 = objs[1][1]

    if len(obj1['analysis_name'].split(' ')) + len(obj2['analysis_name'].split(' ')) <= 2:
        pr = fuzz.ratio(obj1['analysis_name'], obj2['analysis_name'])
    elif obj1['analysis_name'].isdigit() or obj2['analysis_name'].isdigit():
        pr = fuzz.ratio(obj1['analysis_name'], obj2['analysis_name'])
    else:
        pr = fuzz.partial_ratio(obj1['analysis_name'], obj2['analysis_name'])

    if pr >= name_ratio and any([(obj1['isnedra_pi'] == obj2['isnedra_pi']),
                                 (obj1['norm_pi'] == obj2['norm_pi'])]):
        return True

    return False

def _compare_pi(objs):
    gr_pi1 = objs[0][1]['isnedra_pi']
    gr_pi2 = objs[1][1]['isnedra_pi']
    n_pi1 = objs[0][1]['norm_pi']
    n_pi2 = objs[1][1]['norm_pi']

    if any([(gr_pi1 == gr_pi2), (n_pi1 == n_pi2)]):
        return True

    return False

def _get_attrs_ratio(name_ratio, attrs):
    name_1 = attrs[0][1][0]
    name_2 = attrs[1][1][0]
    gr_pi1 = attrs[0][1][1]
    gr_pi2 = attrs[1][1][1]
    n_pi1 = attrs[0][1][2]
    n_pi2 = attrs[1][1][2]

    if len(name_1.split(' ')) + len(name_2.split(' ')) <= 2:
        pr = fuzz.ratio(name_1, name_2)
    elif name_1.isdigit() or name_2.isdigit():
        pr = fuzz.ratio(name_1, name_2)
    else:
        pr = fuzz.partial_ratio(name_1, name_2)

    if pr >= name_ratio and any([(gr_pi1 == gr_pi2), (n_pi1 == n_pi2)]):
        return attrs[0][0], attrs[1][0]


def _compare_polygons_wkt(polygons):
    poly_1 = polygons[0][1]
    poly_2 = polygons[1][1]
    if poly_1['_wkt_'] == poly_2['_wkt_']:
        return polygons[0][0], polygons[1][0]


def _fuzz_similar_ratio(names):
    name1, name2 = names
    if len(name1.split(' ')) + len(name2.split(' ')) <= 2:
        ratio = fuzz.ratio(name1, name2)
    else:
        ratio = fuzz.partial_ratio(name1, name2)

    return ratio

def _group_pi_equal(group_pi_couple):
    return any([(group_pi_couple[0][0] == group_pi_couple[1][0]),
                (group_pi_couple[0][1] == group_pi_couple[1][1])])
