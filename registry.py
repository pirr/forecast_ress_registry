# -*- coding: utf-8 -*-
"""

"""

import re
import numpy as np
import pandas as pd
from collections import OrderedDict


REGISTRY_COLUMNS = OrderedDict([(u'№ строки', 'N'),
                                (u'Актуальность строки', 'actual'),
                                (u'№ изменений', 'N_change'),
                                (u'Операция внесения (добавление, изменение, удаление)',
                                 'change_type'),
                                (u'№ объекта', 'N_objectX'),
                                (u'Признак комплексного', 'complex'),
                                (u'Вид документа регистрации1)', 'doc_type'),
                                (u'Наличие паспорта ГКМ', 'obj_with_gkm'),
                                (u'Орган регистрации (ТФИ, РГФ, ВСЕГЕИ, ЦНИГРИ, Роснедра, Минприроды, ГСЭ)',
                                 'organ_regs'),
                                (u'Номер документа', 'doc_num'),
                                (u'Дата регистрации', 'doc_date'),
                                (u'Год регистрации (для сортировки)', 'doc_date_num'),
                                (u'№ объекта в документе регистрации',
                                 'obj_num_in_doc'),
                                (u'Федеральный округ', 'fed_distr'),
                                (u'Субъект РФ', 'subj_distr'),
                                (u'Административный район', 'adm_distr'),
                                (u'Лист м-ба 1000', 'list_1000'),
                                (u'Лист м-ба 200 (араб.)', 'list_200'),
                                (u'Вид объекта2)', 'geol_type_obj'),
                                (u'Название объекта', 'name_obj'),
                                (u'Фонд недр (Р-распред., НР-нераспред.)', 'fund'),
                                (u'Вид пользования недрами (ГИН/Р+Д/ГИН+Р+Д)', 'use_type'),
                                (u'Группа ПИ в госпрограмме3)', 'gover_type_pi'),
                                (u'ПИ (перечень для объекта)', 'pi'),
                                (u'Название нормализ.', 'norm_pi'),
                                (u'Название ПИ по ГБЗ', 'gbz_pi'),
                                (u'Группа ПИ ИС недра', 'isnedra_pi'),
                                (u'Ед. измерения ПИ', 'unit_pi'),
                                (u'P3', 'P3_cat'),
                                (u'P2', 'P2_cat'),
                                (u'P1', 'P1_cat'),
                                (u'С2', 'C2_res'),
                                (u'Без категор.', 'none_cat'),
                                (u'Запасы ABC1', 'ABC_res'),
                                (u'Признак наличия ресурсных оценок', 'res_exist'),
                                (u'Наличие прогнозных ресурсов', 'cat_avaibil'),
                                (u'Признак наличия запасов', 'res_avaibil'),
                                (u'Вид документа апробации (протокол, отчет)',
                                 'probe_doc_type'),
                                (u'Номер', 'probe_doc_num'),
                                (u'Дата', 'probe_doc_date'),
                                (u'Орган апробации', 'probe_doc_organ'),
                                (u'Территория органа апробации', 'probe_organ_subj'),
                                (u'№ в таблице координат для полигонов', 'N_poly_table'),
                                (u'Вид координат (Т-точка, П-полигон)', 'coord_type'),
                                (u'Площадь, км2', 'area'),
                                (u'Координата центра X', 'lon'),
                                (u'Координата центра Y', 'lat'),
                                (u'Источник координат4)', 'coord_source'),
                                (u'Входимость в лицензионыый участок', 'license_area'),
                                (u'Достоверность координат', 'coord_reliability'),
                                (u'Координаты треб. проверки', 'coord_for_check'),
                                (u'Данные о районе (для определения координат)',
                                 'territory_descript'),
                                (u'Другие документы об объекте (вид документа, №, год, стадия ГРР, авторы, организация)',
                                 'other_source'),
                                (u'Рекомендуемые работы (оценка ПР, апробация ПР, в фонд заявок, поиски, оценка и др.)',
                                 'recommendations')])

INVERT_REGISTRY_COLUMNS = OrderedDict(
    [(v, k) for k, v in REGISTRY_COLUMNS.items()])

actual_cols = ('_id', '_rev', 'id_reg', 'filename')

# name_patterns = pd.read_csv('D://Smaga//bitbucket//obj_creator//dict//pattern_for_replace.csv', delimiter=';', encoding='cp1251')

class RegistryExc(Exception):
    pass

class RegistryFormatter:
    u'''
        верификация и форматирование реестра для импорта в БД
     '''

    pi_dict = pd.read_csv(u'dict//pi2.csv', delimiter=';', encoding='cp1251')

    def __init__(self, registry_df, registry_cols_dict, ):
        self.registry = registry_df
        self.cols = registry_cols_dict
        self.errors = dict()

    # сбор ошибок верификации реестра
    def _append_errors(self, err_name, err_str):
        if err_name in self.errors:
            self.errors[err_name].extend(err_str)
        else:
            self.errors[err_name] = [err_str]

    # проверка наличия ошибок
    def check_errors(self):
        if self.errors:
            self.errors = '\n'.join(str(k) + ':' + str(v) for k,v in self.errors.items())
            raise RegistryExc(self.errors)

    # удаление переносов и других непробельных символов в названии колонок
    # реестра
    def columns_strip(self):
        pattern = re.compile(r'\s+')
        self.registry.columns = [pattern.sub(
            ' ', c) for c in self.registry.columns]

    # проверка наличия колонок, если отсутсвуют то записать их в
    # соотвествующую ошибку
    def check_columns(self):
        none_cols = [c for c in self.cols.keys()
                     if c not in self.registry.columns]
        if none_cols:
            self._append_errors(
                u'В реестре отсутствуют колонки', ', '.join(none_cols))

    # обновление названий колонок для БД
    def update_column_names_for_db(self):
        self.registry.columns = [self.cols.values()]

    # округление чисел с плавающей точкой
    def fix_float(self):
        for col in self.registry.columns:
            if self.registry[col].dtype == np.float64:
                self.registry[col] = np.round(self.registry[col], 6)

    # ошибки координат
    @staticmethod
    def check_coord(coord):
        normal_coord_list = []
        for c in coord:
            coord_is_normal = True
            try:
                _, str_coord_decim = str(c).split('.')
            except ValueError:
                return u'err'
            len_str_coord_decim = len(str_coord_decim)
            if len_str_coord_decim < 3:
                coord_is_normal = False
            elif len(set(str_coord_decim)) <= len_str_coord_decim / 2:
                coord_is_normal = False
            normal_coord_list.append(coord_is_normal)
        if not any(normal_coord_list):
            return u'err'
        return u'ok'

    def prepare_coord(self):
        self.registry[u'coord_checked'] = self.registry.apply(
            lambda row: self.check_coord([row['lon'], row['lat']]) if not pd.isnull(row['lon']) else np.nan, axis=1)
        return u'coords with err: {}'.format(self.registry.loc[self.registry['coord_checked'] == u'err', u'N'])

        # self.registry['coord'] = self.registry.loc[
        #     ~pd.isnull(self.registry['lon']), ['lon', 'lat']]
        # gkm_coords['coord'] = gkm_coords.apply(lambda coord: )

    @property
    def prep_n_poly_column(self):
        for n in self.registry['N_poly_table']:
            try:
                str(n).lower()
            except UnicodeEncodeError as e:
                print n
        self.registry['N_poly_table'] = self.registry['N_poly_table'].astype(str).str.lower()

    @property
    def prep_pi_column(self):
        self.registry['norm_pi'] = self.registry['norm_pi'].str.lower()

    @property
    def check_group_pi(self):
        group_pi_series = self.registry.loc[pd.isnull(self.registry['group_pi']), 'norm_pi']
        if not group_pi_series.empty:
            group_pi_series.drop_duplicates().to_csv('unknow_pi.csv', sep=';', encoding='cp1251')
            raise RegistryExc('Unknow pi in unknow_pi.csv file')

    @property
    def __merg_pi(self):
        self.registry = pd.merge(self.registry, self.pi_dict, left_on='norm_pi', right_on='pi', how='left')


    def format(self, grand_taxons=False):
        self.columns_strip()
        self.check_columns()
        self.check_errors()
        self.fix_float()
        self.update_column_names_for_db()
        self.prep_n_poly_column
        # self.prep_pi_column
        # self.__merg_pi
        # self.check_group_pi
        self.prepare_coord()
        if not grand_taxons:
            self.registry = self.registry[~self.registry['geol_type_obj'].str.lower().isin([u'кт', u'КТ', u'KT', u'kt',
                                                                                            u'мз', u'пгхо', u'пп',
                                                                                            u'рз', u'рп', u'рр', u'ру'])]
