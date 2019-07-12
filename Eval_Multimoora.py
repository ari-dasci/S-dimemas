import pandas as pd
import numpy as np


#Se carga el archivo csv con los datos de las evaluaciones
datos = pd.read_csv('La_noche_2019_normalizado.csv', delimiter=',')
#print(datos)

#Se crea un array con el número total que cada criterio fue evaluado tpc-> total por criterio
tpc = datos.iloc[:,7:].count()
#print(tpc)

#Se obtiene el número total de criterios evaluados dentro del DataFrame. total -> Todos los criterios evaluados.
total = tpc.sum()

#Se crea un array con los pesos que tendrá cada criterio. pesos = tpc/total
pesos = tpc/total
#print(pesos)
#la suma de los pesos debe ser igual a 1
#print(sum(pesos))

#___________________________________________________________________________________________________

#h = ((((datos.iloc[:,6:].abs()-1)*12)/2)+1)
#h = ((((datos.iloc[:,6:].abs()-1)*12)/4)+1)
#h = ((((datos.iloc[:,6:].abs()-1)*12)/6)+1)

#___________________________________________________________________________________________________

#Se agrupan todas las valoraciones por provincia y por subevento
event1 = datos[(datos.PROVINCIA == 'Granada') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event2 = datos[(datos.PROVINCIA == 'Granada') & (datos.SUBEVENTO == 'TallerUrano')]
event3 = datos[(datos.PROVINCIA == 'Sevilla') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event4 = datos[(datos.PROVINCIA == 'Jaen') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event5 = datos[(datos.PROVINCIA == 'Cordoba') & (datos.SUBEVENTO == 'TallerMonuMAI')]
#print(event1.iloc[:,7:])

#se calculan los promedios de cada criterio.
med1 = event1.iloc[:,7:].mean()
med2 = event2.iloc[:,7:].mean()
med3 = event3.iloc[:,7:].mean()
med4 = event4.iloc[:,7:].mean()
med5 = event5.iloc[:,7:].mean()
#print(med5)


#d = np.sqrt(med1)
#print(d)



