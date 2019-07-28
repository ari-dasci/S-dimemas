import pandas as pd
import numpy as np


#Se carga el archivo csv con los datos de las evaluaciones
#Éste archivo ya está normalizado, solo está pendiente el paso para normalizar.
datos = pd.read_csv('La_noche_2019_normalizado.csv', delimiter=',')
#print(datos)

#___________________________________________________________________________________________________
#Pendiente la normalización de los datos a una sola escala
#pero se usaría estas operaciones
#escala3 = ((((datos.iloc[:,6:].abs()-1)*12)/2)+1)
#escala5 = ((((datos.iloc[:,6:].abs()-1)*12)/4)+1)
#escala7 = ((((datos.iloc[:,6:].abs()-1)*12)/6)+1)

#Pendiente el vector de beneficio/costo
#___________________________________________________________________________________________________

#Se crea un array con el número total de veces que cada criterio fue evaluado tpc-> total por criterio
tpc = datos.iloc[:,7:].count()
#print(tpc)

#Se obtiene el número total de veces que todos los criterios fueron evaluados dentro del DataFrame.
# total -> Todos los criterios evaluados.
total = tpc.sum()

#Se crea un array con los pesos que tendrá cada criterio. pesos = tpc/total
#En éste punto solo se está considerando un paraámetro para los pesos.
pesos = tpc/total
#print(pesos)
#la suma de los pesos debe ser igual a 1
#print(sum(pesos))



#Se agrupan todas las valoraciones por provincia y por subevento
#Para el ejemplo, ya se tienen las provincias y subeventos. Que se puede cambiar a evento y actividades.
event1 = datos[(datos.PROVINCIA == 'Granada') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event2 = datos[(datos.PROVINCIA == 'Granada') & (datos.SUBEVENTO == 'TallerUrano')]
event3 = datos[(datos.PROVINCIA == 'Sevilla') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event4 = datos[(datos.PROVINCIA == 'Jaen') & (datos.SUBEVENTO == 'TallerMonuMAI')]
event5 = datos[(datos.PROVINCIA == 'Cordoba') & (datos.SUBEVENTO == 'TallerMonuMAI')]
#print(event1.iloc[:,7:])
#Crea un nuevo array con la concatenación de las cadenas que filtraron a cada evento
#Para que sirvan como las etiquetas de cada evento y actividad, mas adelante se usa para etiquetar los rankings
nwcol = pd.Series(['Granada-Taller MonuMAI','Granada-Taller Urano','Sevilla-Taller MonuMAI','Jaen-Taller MonuMAI',
                   'Cordoba-Taller MonuMAI'], name='subvento')

#se hace el filtro solo por criterios y se calculan las medias aritméticas de cada criterio por cada subevento.
med1 = event1.iloc[:,7:].mean()
med2 = event2.iloc[:,7:].mean()
med3 = event3.iloc[:,7:].mean()
med4 = event4.iloc[:,7:].mean()
med5 = event5.iloc[:,7:].mean()
#print(med5)

#se calcula el cuadrado de cada elemento del array, que son las medias de cada subevento.
#ejemplo med1[1:1]**
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

#Se calcula la Matriz x_ij^* , tomando cada valor del criterio y dividiendolo entre la
#raiz cuadrada de la suma de los cuadarados de los valores de ese criterio.
#valor-criterio/suma de todos los valor-criterio^2
mxij = med1/ np.sqrt(s_cuad)
#Se genera la matriz mxij concatenando los demás eventos utilizando la fórmula anterior
mxij = pd.concat([mxij,med2/np.sqrt(s_cuad),med3/np.sqrt(s_cuad),med4/np.sqrt(s_cuad),med5/np.sqrt(s_cuad)], axis = 1)
#print(mxij)

#se genera la transpuesta de la matriz (tmxij) para que coincidan las colunmas del array de pesos con la matriz mxij
tmxij = mxij.transpose()

#Se ontiene una nueva matriz (wxij) con cada uno de los criterios multiplicado por su peso.
wxij = tmxij * pesos


#Para obtener el primer ranking (RS), se suman los criterios de beneficio o restan los criterios de costo por evento.
#En éste caso, todos son positivos, si existiera un negativo, ese valor se restaría.
#Aqui se aplicaría el array de Beneficio / Costo, que está pendiente por implementar.
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
#Aqui tambien se utilizaría el vector de Beneficio / Costo para obtener los máximos y mínimos. Falta implementar
maximos = mxij.transpose().max()
#Si existieran criterios negativos, entonces el array seria mixto, donde los criterios positivos son los máximos y los
#criterios negativos los mínimos.



#Para calcular el array de referencia, tomaremos los valores máximos de cada criterio que es beneficio y
#los mínimos de costo, ayudandonos del vector beneficio, costo, y los multiplicaremos por el peso que cada criterio tiene.
wrj = pesos * maximos

#Para calcular el segundo ranking (RPA), a cada criterio, se le resta el valor del maximo/minimo multiplicado por el peso (wrj)
rpa = wxij - wrj
#Se genera un array con los valores máximos (Se aplica valor absoluto al resultado de la operación anterior)
#de cada evento
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



#Se toma cada criterio de la matriz mxij, y se eleva a una potencia, que es igual al peso de cada criterio.
#Es decir criterio ^ peso_criterio.
xijpw = mxij.transpose() ** pesos
#print(xijpw)

#Para el tercer ranking(fmf), se utiliza la matrix generada "xijpw", por cada evento, los criterios positivos,
#se multiplican, y los criterios negativos se dividen
#Ejemplo: crit_pos * crit_pos / crit_neg * crit_pos * crit_pos / crit_neg.
#En éste ejemplo todos son positivos, por eso todos se multiplican
#Se utiliza el array para que los criterios de beneficio se multipliquen y los de costo se dividan.
# Igual que en el paso anterior, se toma cada elemento y se eleva a la potencia que le corresponde a su criterio.
#Falta implementar este paso.
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

#-----------------------------------------------------------------------------------------------------------------------
#Para sacar la mejor opcion utilizando el método Rank Position Method (RPM), se haran los siguientes cálculos:
#Primero tomamos todos los rankings por evento (RS, RPA, FMF), y creamos una matriz.

rpm = pd.concat([rs11,rpa222,fmf33], axis = 0)

#Se crean una matriz con los datos filtrados por subevento, la matriz tiene el índice y el ranking que obtuvo por ese
#índice
x = rpm[(rpm['subvento']== nwcol[0])]
x1 = rpm[(rpm['subvento']== nwcol[1])]
x2 = rpm[(rpm['subvento']== nwcol[2])]
x3 = rpm[(rpm['subvento']== nwcol[3])]
x4 = rpm[(rpm['subvento']== nwcol[4])]
#print(x)

#Se calculan los nuevos índices tomando la posición que cada subevento tuvo en los ranking rs, rpa, fmf
xx = 1/(1/x.iloc[0,2] + 1/x.iloc[1,2] + 1/x.iloc[2,2])
xx1 = 1/(1/x1.iloc[0,2] + 1/x1.iloc[1,2] + 1/x1.iloc[2,2])
xx2 = 1/(1/x2.iloc[0,2] + 1/x2.iloc[1,2] + 1/x2.iloc[2,2])
xx3 = 1/(1/x3.iloc[0,2] + 1/x3.iloc[1,2] + 1/x3.iloc[2,2])
xx4 = 1/(1/x4.iloc[0,2] + 1/x4.iloc[1,2] + 1/x4.iloc[2,2])
#print(xx)

#Se crea un nuevo dataframe con los índices de cada sub evento
rpm1 = pd.DataFrame({
    'subevento':[nwcol[0],nwcol[1],nwcol[2],nwcol[3],nwcol[4]],
    'indice':[xx,xx1,xx2,xx3,xx4]},
    columns = ['subevento','indice']
)

#Se ordenan los elementos de la matriz de menor a mayor
rpm1 = rpm1.sort_values(by = 'indice', ascending = True)
#Con ésta funcíon se reordenan los índices para que empiece de 0
rpm1 = rpm1.reset_index(drop=True)
#Se le agrega la columna ranking, con base en los índices reordenados
rpm1.insert(len(rpm1.columns),'ranking RPM', range(len(rpm1)))
#Se le suma 1 para que el ranking empice en 1.
rpm1['ranking RPM'] = rpm1['ranking RPM']+1
#print(rpm1)

#-----------------------------------------------------------------------------------------------------------------------
#Para sacar la mejor opción utilizando el método Improved Borda Rule(IMB) se realizan las siguientes operaciones:
#Primero se crea una nueva matriz con todos los índices generados en RS(y_i), RPA(z_i), FMF(u_i).

imb = pd.DataFrame({
    'subevento':[nwcol[0],nwcol[1],nwcol[2],nwcol[3],nwcol[4]],
    'y_i':[x.iloc[0,0],x1.iloc[0,0],x2.iloc[0,0],x3.iloc[0,0],x4.iloc[0,0]],
    'z_i':[x.iloc[1,0],x1.iloc[1,0],x2.iloc[1,0],x3.iloc[1,0],x4.iloc[1,0]],
    'u_i':[x.iloc[2,0],x1.iloc[2,0],x2.iloc[2,0],x3.iloc[2,0],x4.iloc[2,0]]},
    columns = ['subevento','y_i','z_i','u_i']
)

#Se elevan al cuadrado todos los elementos de los índices
vimb2 = np.square(imb.iloc[0:,1:])
#Se obtiene un array con la suma de los cuadrados de cada índice
vimb22 =  vimb2.sum(axis=0)
#Se crea la matriz normalizada,tomando cada índice y dividiendolo entre la
# raiz cuadrada de la suma de los cuadarados de ese índice. indice/suma de todos los indice^2
imb222 = imb
imb222['y_i'] = imb['y_i']/np.sqrt(vimb22['y_i'])
imb222['z_i'] = imb['z_i']/np.sqrt(vimb22['z_i'])
imb222['u_i'] = imb['u_i']/np.sqrt(vimb22['u_i'])
#print(imb222)

#Se asigna a una variable el número de posiciones que integran los rankings, que es igual al número de subeventos que
#son evaluados
m = nwcol.count()
#Se calcula el valor del denominador en la fórmula para calcular el IMB
m1 = m*(m+1)/2

#Se estraen los rankings numericos  de cada evento. Ejemplo: x.iloc[0,2], x.iloc[1,2], x.iloc[2,2]
#Se extraen los índices de cada evento. Ejemplo: imb222.iloc[0,1], imb222.iloc[0,2], imb222.iloc[0,3]
y = imb222.iloc[0,1] * (m-x.iloc[0,2]+1)/m1 - imb222.iloc[0,2] * (x.iloc[1,2])/m1 + imb222.iloc[0,3] * (m-x.iloc[2,2]+1)/m1
y1 = imb222.iloc[1,1] * (m-x1.iloc[0,2]+1)/m1 - imb222.iloc[1,2] * (x1.iloc[1,2])/m1 + imb222.iloc[1,3] * (m-x1.iloc[2,2]+1)/m1
y2 = imb222.iloc[2,1] * (m-x2.iloc[0,2]+1)/m1 - imb222.iloc[2,2] * (x2.iloc[1,2])/m1 + imb222.iloc[2,3] * (m-x2.iloc[2,2]+1)/m1
y3 = imb222.iloc[3,1] * (m-x3.iloc[0,2]+1)/m1 - imb222.iloc[3,2] * (x3.iloc[1,2])/m1 + imb222.iloc[3,3] * (m-x3.iloc[2,2]+1)/m1
y4 = imb222.iloc[4,1] * (m-x4.iloc[0,2]+1)/m1 - imb222.iloc[4,2] * (x4.iloc[1,2])/m1 + imb222.iloc[4,3] * (m-x4.iloc[2,2]+1)/m1
#print(y1)

#Se crea un data frame con los subeventos y los nuevos índices generados con las operaciones anteriores.
rimb = pd.DataFrame({
    'subevento':[nwcol[0],nwcol[1],nwcol[2],nwcol[3],nwcol[4]],
    'indice':[y,y1,y2,y3,y4]},
    columns = ['subevento','indice']
)

#Se ordenan los elementos de la matriz de mayor a menor
rimb = rimb.sort_values(by = 'indice', ascending = False)
#Con ésta funcíon se reordenan los índices para que empiece de 0
rimb = rimb.reset_index(drop=True)
#Se le agrega la columna ranking, con base en los índices reordenados
rimb.insert(len(rimb.columns),'ranking IMB', range(len(rimb)))
#Se le suma 1 para que el ranking empice en 1.
rimb['ranking IMB'] = rimb['ranking IMB']+1
#print(rimb)
#-----------------------------------------------------------------------------------------------------------------------
print('Rank Position Method (RPM)')
print(rpm1)
print()
print( 'Improved Borda Rule(IMB)')
print(rimb)
#-----------------------------------------------------------------------------------------------------------------------
#Se exporta el DataFrame con el índice Rank Position Method (RPM) a un archivo csv
rpm1.to_csv('indice_RPM.csv', header = True, index = False)
#Se exporta el DataFrame con el índice Improved Borda Rule(IMB) a un archivo csv
rimb.to_csv('indice_IMB.csv', header = True, index = False)