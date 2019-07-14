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
#Pendiente
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
#Para que sirvan como las etiquetas de cada subevento
nwcol = pd.Series(['Granada-Taller MonuMAI','Granada-Taller Urano','Sevilla-Taller MonuMAI','Jaen-Taller MonuMAI','Cordoba-Taller MonuMAI'], name='subvento')

#se hace el filtro por criterios y se calculan las medias aritméticas de cada criterio por cada subevento.
med1 = event1.iloc[:,7:].mean()
med2 = event2.iloc[:,7:].mean()
med3 = event3.iloc[:,7:].mean()
med4 = event4.iloc[:,7:].mean()
med5 = event5.iloc[:,7:].mean()
#print(med5)

#se calcula el cuadrado de cada elemento del array de las medias de cada subevento. e
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
mxij = pd.concat([mxij,med2/np.sqrt(s_cuad),med3/np.sqrt(s_cuad),med4/np.sqrt(s_cuad),med5/np.sqrt(s_cuad)], axis = 1)
#print(mxij)

#se genera la transpuesta de la matriz (tmxij) para que coincidan las colunmas del array de pesos con la matriz mxij
tmxij = mxij.transpose()

#Se ontiene una nueva matriz (wxij) con cada uno de los criterios multiplicado por su peso.
wxij = tmxij * pesos


#Para obtener el primer ranking (RS), se suman/restan los valores de cada criterio por evento.
#En éste caso, todos son positivos, si existiera un negativo, ese valor se restaría.
#ejemplo crit_pos + crit_pos - crit_neg + crit_pos + crit_pos - crit_neg
rs = wxij.sum(axis = 1)
#Se el agrega una nueva columna con el array de los filtros.
rs1 = pd.concat([rs, nwcol], axis=1)
#Se ordenan los eventos de mayor a menor, por los datos de la variable ranking1
rs11 = rs1.sort_values(by = 0, ascending= False)
#Con ésta funcíon se reordenan los índices para que empiece de 0
rs11 = rs11.reset_index(drop=True)
#Se le agrega la columna ranking, con base en los índices reordenados
rs11.insert(len(rs11.columns),'ranking', range(len(rs11)))
#Se le suma 1 para que el ranking empice en 1.
rs11['ranking'] = rs11['ranking']+1
#print(rs11)


#De la matriz mxij se extraen los valores máximos y mínimos, con el se crea un vector. En éste caso todos son positivos.
maximos = mxij.transpose().max()
#Si existieran criterios negativos, entonces el array seria mixto, donde los criterios positivos son los máximos y los
#criterios negativos mínimos.
#minimos = mxij.transpose().min()
#print(minimos)


#Se multiplica el máximo/minimo de cada criterio por su peso (peso)
wrj = pesos * maximos

#Para calcular el segundo ranking (RPA), a cada criterio, se le resta el valor del maximo/minimo multiplicado por el peso (wrj)
rpa = wxij - wrj
#Se genera un array con los valores máximos de cada evento
rpa2 = rpa.abs().transpose().max()
#Se el agrega una nueva columna con el array de los filtros.
rpa22 = pd.concat([rpa2, nwcol], axis=1)
#Se ordenan los eventos de menor a mayor, por los datos de la variable r22
rpa222 = rpa22.sort_values(by = 0, ascending= True)
#Con ésta funcíon se reordenan los índices para que empiece de 0
rpa222 = rpa222.reset_index(drop=True)
#Se le agrega la columna ranking, con base en los índices reordenados
rpa222.insert(len(rpa222.columns),'ranking', range(len(rpa222)))
#Se le suma 1 para que el ranking empice en 1.
rpa222['ranking'] = rpa222['ranking']+1
#print(rpa222)



#Se toma cada criterio de la matriz mxij, y se eleva a una potencia que es igual al peso de cada criterio.
#Es decir criterio ^ peso_criterio.
xijpw = mxij.transpose() ** pesos
#print(xijpw)

#Para el tercer ranking(fmf), por cada evento, los criterios positivos, se multiplican, y los criterios negativos se dividen
#Ejemplo: crit_pos * crit_pos / crit_neg * crit_pos * crit_pos / crit_neg.
#En éste ejemplo todos son positivos, por eso todos se multiplican
fmf = np.prod(xijpw.transpose())
#Se el agrega una nueva columna con el array de los filtros.
fmf3 = pd.concat([fmf, nwcol], axis=1)
fmf33 = fmf3.sort_values(by = 0, ascending= False)
#Con ésta funcíon se reordenan los índices para que empiece de 0
fmf33 = fmf33.reset_index(drop=True)
#Se le agrega la columna ranking, con base en los índices reordenados
fmf33.insert(len(fmf33.columns),'ranking', range(len(fmf33)))
#Se le suma 1 para que el ranking empice en 1.
fmf33['ranking'] = fmf33['ranking']+1
#print(fmf33)

#Para sacar la mejor opcion utilizando el método Rank Position Method (RPM), se haran los siguientes cálculos
#Primero tomamos todos los rankings por evento (RS, RPA, FMF), sin ordenar y creamos una matriz.
#rpm = rs
rpm4 = pd.concat([rs11,rpa222,fmf33], axis = 0)

x = rpm4[(rpm4['subvento']== nwcol[0])]
x1 = rpm4[(rpm4['subvento']== nwcol[1])]
x2 = rpm4[(rpm4['subvento']== nwcol[2])]
x3 = rpm4[(rpm4['subvento']== nwcol[3])]
x4 = rpm4[(rpm4['subvento']== nwcol[4])]

#xx = 1/(1/x.iloc[0,2] + 1/x.iloc[1,2] + 1/x.iloc[2,2])
#print(rpm4)
#rpm = {'subevento'}

print(x)
x5 = 1/(1/x.iloc[0,2] + 1/x.iloc[1,2] + 1/x.iloc[2,2])
#a = 1/(1/rpm4[0] + 1/rpm4[1] + 1/rpm4[2])
print(x5)
#print(x1.iloc[0,2])