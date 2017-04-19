# coding=utf-8
import pandas as pd
from registry import RegistryFormatter, REGISTRY_COLUMNS

path = u'reestr.xls'
data = pd.read_excel(path, sheetname=u'Реестр', skiprows=1)



group_by_pi = data.groupby()