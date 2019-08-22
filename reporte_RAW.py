import numpy as np
import math as ma
import pandas as pd

#import matplotlib.pyplot as plt

#Se carga el archivo csv con los datos de las evaluaciones
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

#Primero se realizan un DataFrame con el conteo general del MACROEVENTO, el número total de evaluaciones, y los rangos de
#edad en los que se encuentra distribuido el total.
fl = {'EVENTO': datos1['EVENTO'].unique(), 'EVALUACIONES': datos1['EVENTO'].count(), 'MENOR de 15':(datos1.EDAD < 15).sum(),
      'ENTRE 15 y 34': ((datos1.EDAD >= 15) & (datos1.EDAD < 34)).sum(), 'ENTRE 35 y 69': ((datos1.EDAD >= 35) & (datos1.EDAD < 69)).sum(),
      'MAYOR DE 70': (datos1.EDAD >= 70).sum()}
filtroA = pd.DataFrame(fl)
#Después se crea un DataFrame con los datos de evaluaciones clasificados por género
filtroA01 = datos1.groupby(['EVENTO', 'GENERO'])['GENERO'].size().reset_index(name='TOTALES')
#print(filtroA)
#filtroA y filtroA01 se pueden exportar como csv, y se pueden usar para generar gráficas referentes al MACROEVENTO



#Se crea un DataFrame con el conteo de evaluaciones por PROVINCIA
fl2 = datos1['PROVINCIA'].value_counts()
filtroB = pd.DataFrame(fl2)
#Despues se realiza un conteo por rangos de edad por cada PROVINCIA
data1 = datos1.groupby(['PROVINCIA'])
datos_guardar1 = []
for nombre, datos in data1:
  lt_15 = (datos.EDAD < 15).sum()
  bt_15_34 = (datos.EDAD < 34).sum() - lt_15
  bt_35_69 = (datos.EDAD < 69).sum() - bt_15_34
  gt_69 = (datos.EDAD > 69).sum()
  datos_guardar1.append([nombre,len(datos),lt_15,bt_15_34, bt_35_69, gt_69])
new_data1 = pd.DataFrame(datos_guardar1,columns = ['PROVINCIA',"NUM EVAL","Menores de 15","Entre 15 y 34","Entre 35 y 69","Mayor de70"])
#Se agrega el conteo por GENERO en un DataFrame diferente,
filtro05 = datos1.groupby(['PROVINCIA','GENERO'])['GENERO'].size().reset_index(name='TOTALES')

#Ya que se tienen los DataFrame por PROVINCIA, ahora se realizarán por SUBEVENTO, primero con los totales de las evaluaciones
filtro02 = datos1.groupby(['PROVINCIA','SUBEVENTO'])['SUBEVENTO'].size().reset_index(name='TOTALES')
#Despues se realiza un conteo por rangos de edad por cada SUBEVENTO
data = datos1.groupby(['PROVINCIA','SUBEVENTO'])
datos_guardar = []
for nombre, datos in data:
  lt_15 = (datos.EDAD<15).sum()
  bt_15_34 = (datos.EDAD < 34).sum() - lt_15
  bt_35_69 = (datos.EDAD < 69).sum() - bt_15_34
  gt_69 = (datos.EDAD > 69).sum()
  datos_guardar.append([nombre, len(datos), lt_15, bt_15_34, bt_35_69, gt_69])
new_data = pd.DataFrame(datos_guardar,columns = ['SUBEVENTO',"NUM EVAL","Menores de 15","Entre 15 y 34","Entre 35 y 69","Mayor de70"])
#Se agrega el conteo por GENERO en un DataFrame diferente,
filtro03 = datos1.groupby(['PROVINCIA','SUBEVENTO','GENERO'])['GENERO'].size().reset_index(name='TOTALES')
#print(filtro03)
