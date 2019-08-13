import numpy as np
import math as ma
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt

#Se carga el archivo csv con los datos de las evaluaciones
#Éste archivo ya está normalizado, solo está pendiente el paso para normalizar.
datos1 = pd.read_csv('normalizados.csv', delimiter=',')
#print(datos1)

#Creamos una tabla pivote para que se agrupen las evalauciones por 'PROVINCIA' y 'SUBEVENTO', son los encabezados en nuestro
#archivo normalizado, que sería "Evento" y "Actividad".
filtro1 = datos1.pivot_table(index=['PROVINCIA', 'SUBEVENTO'])
#Se crean dos columnas que contienen las agrupaciones hechas con el filtro anterior, para obtener cada actividad con su
#respectivo evento relacionado
filtro2 = filtro1.rename_axis(None, axis=1).reset_index()
#Creamos una tabla que contiene cada actividad con su respectivo evento, la utilizaremos como base para las demás operaciones
filtro = filtro2.iloc[:, 0:2]
#print(filtro)

conteo = datos1.iloc[:, :5]
genero = conteo.groupby('GENERO')
print(conteo)
print(genero.describe())


a = len(filtro)
#Calculamos la media aritmética de cada SUBEVENTO (actividad) por criterio
resultado = []
for i in range(a):
    resultado.append(conteo[(conteo.PROVINCIA == filtro.PROVINCIA[i]) & (conteo.SUBEVENTO == filtro.SUBEVENTO[i])].count())

#print(resultado)

datos2 = pd.DataFrame(resultado)
#print(datos2)
datos3 = pd.concat([filtro,datos2], axis=1)
#print(datos3)


#Creo dos arrays, uno con el conteo por evento del número de hombres y otro con el número de mujeres.
#hom = [(event1.GENERO == 'Hombre').sum(),(event2.GENERO == 'Hombre').sum(),
       #(event3.GENERO == 'Hombre').sum(),(event4.GENERO == 'Hombre').sum(),(event1.GENERO == 'Hombre').sum()]
#muj = [(event1.GENERO == 'Mujer').sum(),(event2.GENERO == 'Mujer').sum(),
       #(event3.GENERO == 'Mujer').sum(),(event4.GENERO == 'Mujer').sum(),(event1.GENERO == 'Mujer').sum()]
#print(muj)

#Si sirve filtro01 = datos1.groupby(['PROVINCIA','SUBEVENTO','GENERO'])['GENERO'].count()

