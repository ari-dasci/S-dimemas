import numpy as np
import math as ma
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt

#Se carga el archivo csv con los datos de las evaluaciones
datos1 = pd.read_csv('normalizados.csv', delimiter=',')
#print(datos1)

total_evento = datos1['EVENTO'].count()
filtro01 = datos1.groupby(['EVENTO', 'GENERO'])['GENERO'].size().reset_index(name='TOTALES')
#print(filtro01)


conteo2 = datos1.iloc[:, :3]
e = datos1['PROVINCIA'].value_counts()
cont_prov = pd.DataFrame(e)
filtro02 = datos1.groupby(['PROVINCIA','SUBEVENTO'])['SUBEVENTO'].size().reset_index(name='TOTALES')
#print(filtro02)




filtro03 = datos1.groupby(['PROVINCIA','SUBEVENTO','GENERO'])['GENERO'].size().reset_index(name='TOTALES')
filtro05 = datos1.groupby(['PROVINCIA','GENERO'])['GENERO'].size().reset_index(name='TOTALES')
filtro06 = datos1.groupby(['EVENTO', 'GENERO'])['GENERO'].size().reset_index(name='TOTALES')
#print(filtro03)
#print(filtro04)
#print(filtro05)
#print(filtro06)

conteo3 = datos1.iloc[:, :4]


resultado = (conteo3.EDAD < 15).sum()

print(resultado)
