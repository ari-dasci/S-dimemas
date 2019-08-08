import numpy as np
import math as ma
import pandas as pd

# Se importa el archivo csv generado en el módulo Normalización, donde las escalas están normalizadas a una sola escala
datos = pd.read_csv('normalizados.csv')

filtro1 = datos.pivot_table(index=['PROVINCIA', 'SUBEVENTO'])
filtro2 = filtro1.rename_axis(None, axis=1).reset_index()
filtro = filtro2.iloc[:, 0:2]
a = len(filtro)

resultado = []
for i in range(a):
    resultado.append(datos[(datos.PROVINCIA == filtro.PROVINCIA[i]) & (datos.SUBEVENTO == filtro.SUBEVENTO[i])].mean())

datos2 = pd.DataFrame(resultado)
datos2 = datos2.drop(['EDAD', 'ESCALA'], axis=1)
datos3 = pd.concat([filtro, datos2], axis=1)
print(datos3)
