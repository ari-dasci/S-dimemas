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
promedioscriterios = pandas.Series(promedioscriterios)#resultado

MCM = numpy.lcm.reduce([int(valor) for valor in promedioscriterios.tolist()])
pesoscriterios = pandas.Series(promedioscriterios)
pesoscriterios = pesoscriterios.apply(lambda x: (MCM / x) * 1)
suma = pesoscriterios.sum()
pesoscriterios = pesoscriterios.apply(lambda x: (1 * x) / suma)#resultado
sumadepesos = pesoscriterios.sum()

pesoparticipacion = {}
for columna in dsPrioridades:
    fdatos = pandas.Series(dsPrioridades[columna])
    pesoparticipacion[columna] = fdatos[fdatos > 0].count()
pesoparticipacion = pandas.Series(pesoparticipacion)
suma = pesoparticipacion.sum()
pesoparticipacion = pesoparticipacion.apply(lambda x: x / suma)#resultado
sumadepesos = pesoparticipacion.sum()

vectorBenCos = {}
for columna in dsEvaluacion:
    fdatos = pandas.Series(dsEvaluacion[columna])
    vectorBenCos[columna] = 1 if fdatos[fdatos == '+'].count() >= fdatos[fdatos == '-'].count() else -1
vectorBenCos = pandas.Series(vectorBenCos)#resultado


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
pesosdimensiones = pandas.Series(pesosdimensiones)
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
ppdimension = pandas.Series(ppdimension)
suma = ppdimension.sum()
ppdimension = ppdimension.apply(lambda x: x / suma)#resultado
sumadepesos = ppdimension.sum()


#multimora inferior
#Utilizando el vector de pesos por importancia y el vector de pesos por participación, se calcula el peso final que tendrá cada criterio
ndp = pesoscriterios.mul(pesoparticipacion)
suma = ndp.sum()
ndp = ndp.apply(lambda x: x / suma)#resultado de Normalización de pesos

#Se calculan las medias aritméticas de cada uno de los criterios( x ), usando el DataSet con los valores inferiores normalizados
filas = numpy.zeros(len(actividad_indices))
columnas = {}
for columna in dsInferior.columns.tolist():
    columnas[columna] = filas

ma = pandas.DataFrame(columnas, actividad_indices.keys())
for columna in dsInferior.columns.tolist():
    for fila in actividad_indices:
        ma[columna][fila] = dsInferior.iloc[actividad_indices[fila]][columna].mean()

#Se elevan al cuadrado cada uno de los elementos (x^2) de la matriz anterior
ma2 = ma.applymap(lambda x: x * x)

#Se suman los cuadrados de cada criterio, y obtenemos un array que usaremos a continuación
suma_ma = {}
for columna in ma2:
    suma_ma[columna] = ma2[columna].sum()

#Se genera una  Matriz ()  tomando de cada criterio la media aritmética por cada evento y se divide entre la raiz cuadrada de la suma de los cuadrados () de cada criterio
mar = pandas.DataFrame(ma)
#print(mar)
for columna in mar.columns.tolist():
    for fila in actividad_indices:
        mar[columna][fila] = mar[columna][fila] / numpy.sqrt( suma_ma[columna] )
#print(mar)

#Se multiplica cada elemento de cada criterio de la Matriz generada, por el peso asociado a ese criterio utilizando el vector de Normalización de pesos. La nueva Matriz es (             )
marm = pandas.DataFrame(mar)
for columna in mar.columns.tolist():
    for fila in actividad_indices:
        mar[columna][fila] = mar[columna][fila] * ndp[columna]
#print(marm)

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se sumen y los de costo se resten
marm_v_bc = pandas.DataFrame( marm)
for columna in marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        marm_v_bc[columna][fila] = marm_v_bc[columna][fila] * vectorBenCos[columna]
print(marm_v_bc)

#Para obtener el primer ranking (RS), se suman los criterios de beneficio o restan los criterios de costo por evento. El índice del ranking es del valor mayor al menor
