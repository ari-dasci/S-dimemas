import pandas as pd
import collections as clns
import numpy as np

"Se carga el archivo csv con los datos de las evaluaciones"
"datos = pd.read_csv('La_noche_2019.csv', index_col=0)"
datos = pd.read_csv('La_noche_2019_normalizado.csv', index_col=0)

"Se calcula el número total de criterios evaluados, de todas las dimensiones y de todos los eventos"
conteo = datos.iloc[:,6:].count(0)
totales = sum(conteo)
print(totales)

"Se crea el vector con los pesos de cada criterio basandonos en el porcentaje de participación de cada criterio con respecto al total"
peso = conteo/totales
"la suma de los pesos debe ser igual a 1"
"print(sum(peso))"

"___________________________________________________________________________________________________"

"h = ((((datos.iloc[:,6:].abs()-1)*12)/2)+1)"
"h = ((((datos.iloc[:,6:].abs()-1)*12)/4)+1)"
"h = ((((datos.iloc[:,6:].abs()-1)*12)/6)+1)"

"___________________________________________________________________________________________________"

"Se agrupan todas las valoraciones por provincia y por subevento"
event1 = datos[(datos.PROVINCIA == 'Granada') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event2 = datos[(datos.PROVINCIA == 'Granada') & (datos.SUBEVENTO == 'TallerUrano')]
event3 = datos[(datos.PROVINCIA == 'Sevilla') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event4 = datos[(datos.PROVINCIA == 'Jaen') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event5 = datos[(datos.PROVINCIA == 'Cordoba') & (datos.SUBEVENTO == 'TallerMonuMAI')]
"print(event3)"

"___________________________________________________________________________________________________"

"Se hace el conteo de los elementos evaluados en cada subevento"
cont_e1 = event1.iloc[:,6:].abs().count(0)
print(cont_e1)





