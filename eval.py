import pandas
import re
import numpy

#carga de los datos
rawData = pandas.read_csv('calculos.csv')#se cargan todos los datos del origen
escala = pandas.Series(rawData['ESCALA'])#se crea una lista solo con las escalas
cEscala = rawData.columns.get_loc('ESCALA')#se determina cual es la columna de las escalas para poder extraer a posterior solamente las evaluaciones de los usuarios
datos = rawData.iloc[:,cEscala+1:]#se extraen unicamente las evaluaciones
actividades = rawData.ACTIVIDAD.unique()#se determinan las diferentes actividades por titulo
actividad_indices = {}
for actividad in actividades:
    indices_escalas = rawData.index[rawData['ACTIVIDAD'] == actividad].tolist()#se obtienen los indices de las escalas correpondientes por actividad
    actividad_indices[actividad] = indices_escalas

#obtención de los dataset por medio de expresiones regulares
dsPrioridades = datos.applymap(lambda x: int(re.search('(\d+)[+-]', x)[1]))
dsEvaluacion = datos.applymap(lambda x: re.search('[+-]', x)[0])
dsInferior = datos.applymap(lambda x: re.search('(\d+):', x)[1])
dsSuperior = datos.applymap(lambda x: re.search(':(\d+)', x)[1])

#print(dsInferior)
#print(dsSuperior)

#normalización
actividad_escala = {}
for actividad in actividades:
    indices_escalas = rawData.index[rawData['ACTIVIDAD'] == actividad].tolist()#se obtienen los indices de las escalas correpondientes por actividad
    escalas = rawData.ESCALA.iloc[indices_escalas].unique()#se obtienes las escalas comotales
    escalas = [i - 1 for i in escalas]#se reduce en uno cada elemento
    actividad_escala[actividad] = numpy.lcm.reduce(escalas)#se crea un diccionario con las escalas para la normalización

for i in dsInferior.index:
    nactividad = rawData.ACTIVIDAD.loc[i]
    MCM = actividad_escala[nactividad]
    Es = escala[i]
    for j in dsInferior:
        Ev = int(dsInferior[j][i])
        dsInferior[j][i] = ((Ev-1)*MCM/(Es-1))+1

        Ev = int(dsSuperior[j][i])
        dsSuperior[j][i] = ((Ev-1)*MCM/(Es-1))+1

#print(dsInferior)
#print(dsSuperior)

#Pesos y vector costo - beneficio
promedioscriterios = {}
for columna in dsPrioridades:
    promedioscriterios[columna] = dsPrioridades[columna].mean()
promedioscriterios = pandas.Series(promedioscriterios.copy())#resultado

MCM = numpy.lcm.reduce([int(valor) for valor in promedioscriterios.tolist()])
pesoscriterios = pandas.Series(promedioscriterios.copy())
pesoscriterios = pesoscriterios.apply(lambda x: (MCM / x) * 1)
suma = pesoscriterios.sum()
pesoscriterios = pesoscriterios.apply(lambda x: (1 * x) / suma)#resultado
sumadepesos = pesoscriterios.sum()

pesoparticipacion = {}
for columna in dsPrioridades:
    fdatos = pandas.Series(dsPrioridades[columna].copy())
    pesoparticipacion[columna] = fdatos[fdatos > 0].count()
pesoparticipacion = pandas.Series(pesoparticipacion.copy())
suma = pesoparticipacion.sum()
pesoparticipacion = pesoparticipacion.apply(lambda x: x / suma)#resultado
sumadepesos = pesoparticipacion.sum()

vectorBenCos = {}
for columna in dsEvaluacion:
    fdatos = pandas.Series(dsEvaluacion[columna].copy())
    vectorBenCos[columna] = 1 if fdatos[fdatos == '+'].count() >= fdatos[fdatos == '-'].count() else -1
vectorBenCos = pandas.Series(vectorBenCos.copy())#resultado


#por dimensiones
dimensiones = dsPrioridades.columns.tolist()
tdimensiones = [item.split(' - ')[0] for item in dimensiones]

dimensiones = []
for dim in tdimensiones:
    if dim not in dimensiones:
        dimensiones.append(dim)

pesosdimensiones = {}
for dimension in dimensiones:
    fdatos = []
    for columna in dsPrioridades:
        if dimension + ' -' in columna:
            fdatos.extend(dsPrioridades[columna].tolist())
    pesosdimensiones[dimension] = sum(fdatos) / len(fdatos)
pesosdimensiones = pandas.Series(pesosdimensiones.copy())
MCM = numpy.lcm.reduce([int(x) for x in pesosdimensiones.tolist()])
pesosdimensiones = pesosdimensiones.apply(lambda x: (MCM / x) * 1)
suma = pesosdimensiones.sum()
pesosdimensiones = pesosdimensiones.apply(lambda x: (1 * x) / suma)#resultado
sumadepesos = pesosdimensiones.sum()


ppdimension = {}
for dimension in dimensiones:
    fdatos = []
    suma = 0
    for columna in dsPrioridades:
        if dimension + ' -' in columna:
            fdatos.extend(dsPrioridades[columna].tolist())
    for i in fdatos:
        if i > 0:
            suma += 1
    ppdimension[dimension] = suma
ppdimension = pandas.Series(ppdimension.copy())
suma = ppdimension.sum()
ppdimension = ppdimension.apply(lambda x: x / suma)#resultado
sumadepesos = ppdimension.sum()


#multimoora inferior
#Utilizando el vector de pesos por importancia y el vector de pesos por participación, se calcula el peso final que tendrá cada criterio
ndp = pesoscriterios.mul(pesoparticipacion)
suma = ndp.sum()
ndp = ndp.apply(lambda x: x / suma)#resultado de Normalización de pesos

#Se calculan las medias aritméticas de cada uno de los criterios( x ), usando el DataSet con los valores inferiores normalizados
filas = numpy.zeros(len(actividad_indices))
columnas = {}
for columna in dsInferior.columns.tolist():
    columnas[columna] = filas

ma = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsInferior.columns.tolist():
    for fila in actividad_indices:
        ma[columna][fila] = dsInferior.iloc[actividad_indices[fila]][columna].mean()
#print(ma)

#Se elevan al cuadrado cada uno de los elementos (x^2) de la matriz anterior
ma2 = ma.applymap(lambda x: x * x)

#Se suman los cuadrados de cada criterio, y obtenemos un array que usaremos a continuación
suma_ma = {}
for columna in ma2:
    suma_ma[columna] = ma2[columna].sum()

#Se genera una  Matriz ()  tomando de cada criterio la media aritmética por cada evento y se divide entre la raiz cuadrada de la suma de los cuadrados () de cada criterio
mar = pandas.DataFrame(ma.copy())
#print(mar)
for columna in mar.columns.tolist():
    for fila in actividad_indices:
        mar[columna][fila] = mar[columna][fila] / numpy.sqrt( suma_ma[columna] )

#Se multiplica cada elemento de cada criterio de la Matriz generada, por el peso asociado a ese criterio utilizando el vector de Normalización de pesos. La nueva Matriz es (             )
marm = pandas.DataFrame(mar.copy())
for columna in marm.columns.tolist():
    for fila in actividad_indices:
        marm[columna][fila] = mar[columna][fila] * ndp[columna]

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se sumen y los de costo se resten
marm_v_bc = pandas.DataFrame(marm.copy())
for columna in marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        marm_v_bc[columna][fila] = marm_v_bc[columna][fila] * vectorBenCos[columna]
#print(marm_v_bc)

#Para obtener el primer ranking (RS), se suman los criterios de beneficio o restan los criterios de costo por evento. El índice del ranking es del valor mayor al menor
RS = {}
for fila in actividad_indices:
    RS[fila] = marm_v_bc.loc[fila].sum()
RS = pandas.Series(RS.copy())
RS = RS.to_frame(name='y_i')
RS['ranking'] = RS.rank(ascending = False)
#print(RS)

#Para el siguiente ranking, se generan dos arrays, uno con los máximos y otros con los mínimos de los criterios en la Matriz (       ), los máximos representan los criterios de beneficios y los mínimos los criterios de costo.
t9 = pandas.DataFrame()
t9 = t9.append( pandas.Series(mar.max(), name='max') )
t9 = t9.append( pandas.Series(mar.min(), name='min') )
#print(t9)

#Para calcular el array de referencia, tomaremos los valores máximos de cada criterio que son beneficio y los mínimos de costo, ayudandonos del Vector Ben(+)/Cost(-), y los multiplicaremos por el peso (    ) que cada criterio tiene.
w_r_r_j = {}
for fila in vectorBenCos.index:
    w_r_r_j[fila] = t9[fila]['max'] * ndp[fila] if vectorBenCos[fila] > 0 else t9[fila]['min'] * ndp[fila]
w_r_r_j = pandas.Series(w_r_r_j.copy())
#print(w_r_r_j)

#Para calcular el segundo ranking (RPA), a cada valor del criterio en la matriz (           ), se le resta el valor de ese criterio en el array de referencia (          ).
RPA = pandas.DataFrame(marm.copy())
for columna in RPA.columns.tolist():
    for fila in RPA.index:
        if RPA[columna][fila] != numpy.nan:
            RPA[columna][fila] = w_r_r_j[columna] - RPA[columna][fila]
#print(RPA)

#Para obtener el segundo ranking (RPA), se toman el valor máximo de cada evento. El índice del ranking es el órden ascendente de los valores anteriores
i_RPA = {}
for fila in actividad_indices:
    i_RPA[fila] = RPA.loc[fila].max()
i_RPA = pandas.Series(i_RPA.copy())
i_RPA = i_RPA.to_frame(name='z_i')
i_RPA['ranking'] = i_RPA.rank()
#print(i_RPA)

#Para el último ranking, se toma cada criterio de la matriz (        ), y se eleva a una potencia, que es igual al peso (      ) de cada criterio. Es decir criterio ^ peso_criterio.
t12 = pandas.DataFrame(mar.copy())
for columna in t12.columns.tolist():
    for fila in t12.index:
        t12[columna][fila] = numpy.power( t12[columna][fila], ndp[columna])
#print(t12)

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se multipliquen y los de costo se dividan. Igual que en el paso anterior, se toma cada elemento de esa nueva Matriz generada y se eleva a la potencia que le corresponde a su criterio.
t13 = pandas.DataFrame(t12.copy())
for columna in t13.columns.tolist():
    for fila in t13.index:
        t13[columna][fila] = numpy.power( t13[columna][fila], vectorBenCos[columna])
#print(t13)

#Para el tercer ranking (FMF), se utiliza nueva la matrix generada, y se multiplican todos los criterios de  cada actividad. Ejemplo Evento1-Actividad1 : criterio1*criterio2*criterio3. . .criterioN. El índice del ranking es el orden de descendente del resultado
FMF = {}
for fila in actividad_indices:
    FMF[fila] = t13.loc[fila].product()
FMF = pandas.Series(FMF.copy())
FMF = FMF.to_frame(name='u_i')
FMF['ranking'] = FMF.rank(ascending = False)
#print(FMF)

#Para utilizar los métodos RPM e IMB y obtener el ranking final de los eventos, necesitaremos los índices y con sus respectivos rankings de cada actividad, obtenidos de los tres métodos anteriores.
#Matriz con los rankings numéricos
rn = pandas.DataFrame()
rn['y_i'] = RS['ranking']
rn['z_i'] = i_RPA['ranking']
rn['u_i'] = FMF['ranking']
print(rn)

#Matriz con los los índices
mi = pandas.DataFrame()
mi['y_i'] = RS['y_i']
mi['z_i'] = i_RPA['z_i']
mi['u_i'] = FMF['u_i']
print(mi)
#Se toman los índices y se suman sus cuadrados
smi2 = {}
smi2['y_i'] = (mi['y_i'] ** 2).sum()
smi2['z_i'] = (mi['z_i'] ** 2).sum()
smi2['u_i'] = (mi['u_i'] ** 2).sum()
smi2 = pandas.Series(smi2)
print(smi2)

#Indices normalizados
IN = pandas.DataFrame(mi.copy())
for columna in IN.columns.tolist():
    for fila in IN.index:
        IN[columna][fila] = IN[columna][fila] / smi2[columna]
print(IN)
