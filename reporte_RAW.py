import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

#Se carga el archivo csv con los datos de las evaluaciones
#Éste archivo ya está normalizado, solo está pendiente el paso para normalizar.
datos1 = pd.read_csv('La_noche_2019_normalizado.csv', delimiter=',')
#print(datos)

prov = datos1['PROVINCIA'].unique()
subev = datos1['SUBEVENTO'].unique()
#print(prov)
#print(subev)

#for i in range(len(prov)):
#    event = datos1[(datos1.PROVINCIA == prov[i])]

#Genero los filtros por provincia y subevento
event1 = datos1[(datos1.PROVINCIA == 'Granada') & (datos1.SUBEVENTO == 'TallerMonuMAI')]
event2 = datos1[(datos1.PROVINCIA == 'Granada') & (datos1.SUBEVENTO == 'TallerUrano')]
event3 = datos1[(datos1.PROVINCIA == 'Sevilla') & (datos1.SUBEVENTO == 'TallerMonuMAI')]
event4 = datos1[(datos1.PROVINCIA == 'Jaen') & (datos1.SUBEVENTO == 'TallerMonuMAI')]
event5 = datos1[(datos1.PROVINCIA == 'Cordoba') & (datos1.SUBEVENTO == 'TallerMonuMAI')]
#print(event1)

#Creo dos arrays, uno con el conteo por evento del número de hombres y otro con el número de mujeres.
hom = [(event1.GENERO == 'Hombre').sum(),(event2.GENERO == 'Hombre').sum(),
       (event3.GENERO == 'Hombre').sum(),(event4.GENERO == 'Hombre').sum(),(event1.GENERO == 'Hombre').sum()]
muj = [(event1.GENERO == 'Mujer').sum(),(event2.GENERO == 'Mujer').sum(),
       (event3.GENERO == 'Mujer').sum(),(event4.GENERO == 'Mujer').sum(),(event1.GENERO == 'Mujer').sum()]
#print(muj)

#Si sirve filtro01 = datos1.groupby(['PROVINCIA','SUBEVENTO','GENERO'])['GENERO'].count()

