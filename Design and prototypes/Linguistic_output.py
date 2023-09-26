import numpy as np
import math as ma
import pandas as pd
#Se definen los diccionarios para transformar el valor numérico en una etiqueta lingüística, en nuestro modelo base tenemos
# 4 conjuntos de etiquetas, las base de la evaluación 3,5,7 y la de normalización 13
et_ling3 = {1: 'POCO', 2: 'NORMAL', 3: 'MUCHO'}
et_ling5 = {1: 'POCO', 2: 'ALGO', 3: 'NORMAL', 4: 'BASTANTE', 5: 'MUCHO'}
et_ling7 = {1: 'NADA', 2: 'MUY POCO', 3: 'REGULAR', 4: 'ALGO', 5: 'NORMAL', 6: 'BASTANTE', 7: 'MUCHÍSIMO'}
et_ling13 = {1: 'POCO', 2: 'POR ENCIMA DE POCO', 3: 'POR DEBAJO DE ALGO', 4: 'ALGO', 5: 'POR ENCIMA DE ALGO',
             6: 'POR DEBAJO DE NORMAL', 7: 'NORMAL', 8: 'POR ENCIMA DE NORMAL', 9: 'POR DEBAJO DE BASTANTE',
             10: 'BASTANTE', 11: 'POR ENCIMA DE BASTANTE', 12: 'POR DEBAJO DE MUCHÍSIMO', 13: 'MUCHÍSIMO'}

# Se importa el archivo csv generado en el módulo Normalización, donde todos los criterios están en una sola escala
datos = pd.read_csv('normalizados.csv')

#Éste paso es para obtener las etiquetas lingüísticas por SUBEVENTO (actividad) sin considerar aún los pesos para cada
#criterio, se puede utilizar para la retroalimentación hacia el evaluador.

#Creamos una tabla pivote para que se agrupen las evalauciones por 'PROVINCIA' y 'SUBEVENTO', son los encabezados en nuestro
#archivo normalizado, que sería "Evento" y "Actividad".
filtro1 = datos.pivot_table(index=['PROVINCIA', 'SUBEVENTO'])
#Se crean dos columnas que contienen las agrupaciones hechas con el filtro anterior, para obtener cada actividad con su
#respectivo evento relacionado
filtro2 = filtro1.rename_axis(None, axis=1).reset_index()
#Creamos una tabla que contiene cada actividad con su respectivo evento, la utilizaremos como base para las demás operaciones
filtro = filtro2.iloc[:, 0:2]
#a = len(filtro)

#Calculamos la media aritmética de cada SUBEVENTO (actividad) por criterio
resultado = []
for i in range(len(filtro)):
    resultado.append(datos[(datos.PROVINCIA == filtro.PROVINCIA[i]) & (datos.SUBEVENTO == filtro.SUBEVENTO[i])].mean())

#Convertimos las listas obtenidas en un nuevo DataFrame con las medias de cada criterio
datos2 = pd.DataFrame(resultado)
#Eliminamos las columnas de EDAD y ESCALA, las cuales no necesitaremos para éste proceso.
datos2 = datos2.drop(['EDAD', 'ESCALA'], axis=1)
#Concatenamos el DataFrame anterior con el DataFrame de los filtros
datos3 = pd.concat([filtro, datos2], axis=1)
#Creamos un nuevo DataFrame con la concatenación de los DataFrame filtro y otro con el cálculo de la media aritmética de
#todos los criterios por evento
datos4 = pd.concat([filtro, datos3.mean(axis=1)], axis=1)
#Al nuevo DataFrame se le agrega una columna llamada BETA, que tendrá los valores redondeados de kas medias aritméticas
datos4['BETA'] = datos4[0].round()
#Se crea otra columna con las mismas características que la anterior, pero ésta nos servirá para sustituir sus valores
datos4['ETIQUETA'] = datos4[0].round()
#Se sustituyen los valores de la columna ETIQUETA, con los del diccionario de acuerdo al valor obtenido en el redondeo
datos4['ETIQUETA'].replace(et_ling13, inplace=True)
#print(datos4)
datos4.to_csv('etiqueta_por_actividad.csv', header=True, index=False)