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
#Crea un nuevo array con la concatenación de las cadenas que filtraron a cada evento
nwcol = pd.Series(['Granada-Taller MonuMAI','Granada-Taller Urano','Sevilla-Taller MonuMAI','Jaen-Taller MonuMAI','Cordoba-Taller MonuMAI'], name='Evento')

#se hace el filtro por criterios y se calculan las medias aritméticas de cada criterio por cada subevento.
med1 = event1.iloc[:,7:].mean()
med2 = event2.iloc[:,7:].mean()
med3 = event3.iloc[:,7:].mean()
med4 = event4.iloc[:,7:].mean()
med5 = event5.iloc[:,7:].mean()
#print(med5)

#se calcula el cuadrado de cada elemento del array de las medias de cada subevento.
e1 = np.square(med1)
e2 = np.square(med2)
e3 = np.square(med3)
e4 = np.square(med4)
e5 = np.square(med5)
#print(e5)

#Se crea un array con la suma de los cuadrados de cada criterio. s_cuad
sc = e1
sc = pd.concat([sc,e2,e3,e4,e5],axis = 1)
s_cuad = sc.sum(axis = 1)
#print(s_cuad)

#Se calcula la Matriz x_ij^* , tomando cada criterio y dividiendolo entre el valor de la
# raiz cuadrada de la suma de los cuadarados de ese criterio. mxij
mxij = med1/ np.sqrt(s_cuad)
mxij = pd.concat([mxij,med2/ np.sqrt(s_cuad),med3/ np.sqrt(s_cuad),med4/ np.sqrt(s_cuad),med5/ np.sqrt(s_cuad)], axis = 1)
#print(mxij)

#se genera la transpuesta de la matriz (tmxij) para que coincidan las colunmas del array de pesos con la matriz mxij
tmxij = mxij.transpose()

#Se ontiene una nueva matriz (wxij) con cada uno de los criterios multiplicado por su peso.
wxij = tmxij * pesos

#Para obtener el primer ranking, se suman/restan los valores de cada criterio por evento.
ranking1 = wxij.sum(axis = 1)
#Se el agrega una nueva columna con el array de los filtros.
r1 = pd.concat([ranking1, nwcol], axis=1)
#Se ordenan los eventos de mayor a menor, por los datos de la variable ranking1
r11 = r1.sort_index(by = 0, ascending= False)


