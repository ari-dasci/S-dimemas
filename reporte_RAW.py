import numpy as np
import math as ma
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt

#Se carga el archivo csv con los datos de las evaluaciones
#Éste archivo ya está normalizado, solo está pendiente el paso para normalizar.
datos1 = pd.read_csv('normalizados.csv', delimiter=',')
#print(datos1)

Total_evento = datos1['EVENTO'].count()
#print(Total_evento)

conteo2 = datos1.iloc[:, :3]

e = datos1['PROVINCIA'].value_counts()
cont_prov = pd.DataFrame(e)
print(cont_prov)




filtro03 = datos1.groupby(['PROVINCIA','SUBEVENTO','GENERO'])['GENERO'].count()
filtro04 = datos1.groupby(['PROVINCIA','SUBEVENTO'])['SUBEVENTO'].count()
filtro05 = datos1.groupby(['PROVINCIA','GENERO'])['GENERO'].count()
filtro06 = datos1.groupby(['EVENTO', 'GENERO'])['GENERO'].count()
print(filtro03)
print(filtro04)
print(filtro05)
print(filtro06)

a = pd.DataFrame(filtro03)
print(a)
