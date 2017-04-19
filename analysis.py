# coding=utf-8

import pandas as pd
import seaborn as sns
from registry import RegistryFormatter, REGISTRY_COLUMNS


pi_cols = [
           u'gbz_pi', u'P3_cat', u'P2_cat',
           u'P1_cat', u'C2_res', u'none_cat', u'ABC_res'
           ]
res_cols = [u'P3_cat', u'P2_cat',
           u'P1_cat', u'C2_res', u'none_cat', u'ABC_res']

path = u'data_analysis//reestr.xls'
data = pd.read_excel(path, sheetname=u'Реестр', skiprows=1)

registry_fmt = RegistryFormatter(data, REGISTRY_COLUMNS)
registry_fmt.format()
xls_registry = registry_fmt.registry

group_by_pi_witobj = xls_registry.groupby([u'gbz_pi', u'N_obj'])[res_cols].max().reset_index()
group_by_pi = group_by_pi_witobj.drop(u'N_obj', axis=1)[pi_cols].groupby(u'gbz_pi').count()

group_count_pi = group_by_pi_witobj.groupby(u'gbz_pi').size()
group_count_pi.plot(kind='bar', subplots=False, figsize=[20, 8])
# print group_by_pi[res_cols].size
# group_by_pi.unstack(level=0).plot(kind='bar', subplots=False, figsize=[20, 10])


