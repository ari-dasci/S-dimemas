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

MMI_ma = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsInferior.columns.tolist():
    for fila in actividad_indices:
        MMI_ma[columna][fila] = dsInferior.iloc[actividad_indices[fila]][columna].mean()
#print(MMI_ma)

#Se elevan al cuadrado cada uno de los elementos (x^2) de la matriz anterior
MMI_ma2 = MMI_ma.applymap(lambda x: x * x)

#Se suman los cuadrados de cada criterio, y obtenemos un array que usaremos a continuación
MMI_suma_ma = {}
for columna in MMI_ma2:
    MMI_suma_ma[columna] = MMI_ma2[columna].sum()

#Se genera una  Matriz ()  tomando de cada criterio la media aritmética por cada evento y se divide entre la raiz cuadrada de la suma de los cuadrados () de cada criterio
MMI_mar = pandas.DataFrame(MMI_ma.copy())
#print(MMI_mar)
for columna in MMI_mar.columns.tolist():
    for fila in actividad_indices:
        MMI_mar[columna][fila] = MMI_mar[columna][fila] / numpy.sqrt( MMI_suma_ma[columna] )

#Se multiplica cada elemento de cada criterio de la Matriz generada, por el peso asociado a ese criterio utilizando el vector de Normalización de pesos. La nueva Matriz es (             )
MMI_marm = pandas.DataFrame(MMI_mar.copy())
for columna in MMI_marm.columns.tolist():
    for fila in actividad_indices:
        MMI_marm[columna][fila] = MMI_mar[columna][fila] * ndp[columna]

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se sumen y los de costo se resten
MMI_marm_v_bc = pandas.DataFrame(MMI_marm.copy())
for columna in MMI_marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        MMI_marm_v_bc[columna][fila] = MMI_marm_v_bc[columna][fila] * vectorBenCos[columna]
#print(MMI_marm_v_bc)

#Para obtener el primer ranking (MMI_RS), se suman los criterios de beneficio o restan los criterios de costo por evento. El índice del ranking es del valor mayor al menor
MMI_RS = {}
for fila in actividad_indices:
    MMI_RS[fila] = MMI_marm_v_bc.loc[fila].sum()
MMI_RS = pandas.Series(MMI_RS.copy())
MMI_RS = MMI_RS.to_frame(name='y_i')
MMI_RS['ranking'] = MMI_RS.rank(ascending = False)
#print(MMI_RS)

#Para el siguiente ranking, se generan dos arrays, uno con los máximos y otros con los mínimos de los criterios en la Matriz (       ), los máximos representan los criterios de beneficios y los mínimos los criterios de costo.
MMI_t9 = pandas.DataFrame()
MMI_t9 = MMI_t9.append( pandas.Series(MMI_mar.max(), name='max') )
MMI_t9 = MMI_t9.append( pandas.Series(MMI_mar.min(), name='min') )
#print(MMI_t9)

#Para calcular el array de referencia, tomaremos los valores máximos de cada criterio que son beneficio y los mínimos de costo, ayudandonos del Vector Ben(+)/Cost(-), y los multiplicaremos por el peso (    ) que cada criterio tiene.
MMI_w_r_r_j = {}
for fila in vectorBenCos.index:
    MMI_w_r_r_j[fila] = MMI_t9[fila]['max'] * ndp[fila] if vectorBenCos[fila] > 0 else MMI_t9[fila]['min'] * ndp[fila]
MMI_w_r_r_j = pandas.Series(MMI_w_r_r_j.copy())
#print(MMI_w_r_r_j)

#Para calcular el segundo ranking (MMI_RPA), a cada valor del criterio en la matriz (           ), se le resta el valor de ese criterio en el array de referencia (          ).
MMI_RPA = pandas.DataFrame(MMI_marm.copy())
for columna in MMI_RPA.columns.tolist():
    for fila in MMI_RPA.index:
        if MMI_RPA[columna][fila] != numpy.nan:
            MMI_RPA[columna][fila] = MMI_w_r_r_j[columna] - MMI_RPA[columna][fila]
#print(MMI_RPA)

#Para obtener el segundo ranking (MMI_RPA), se toman el valor máximo de cada evento. El índice del ranking es el órden ascendente de los valores anteriores
MMI_i_RPA = {}
for fila in actividad_indices:
    MMI_i_RPA[fila] = MMI_RPA.loc[fila].max()
MMI_i_RPA = pandas.Series(MMI_i_RPA.copy())
MMI_i_RPA = MMI_i_RPA.to_frame(name='z_i')
MMI_i_RPA['ranking'] = MMI_i_RPA.rank()
#print(MMI_i_RPA)

#Para el último ranking, se toma cada criterio de la matriz (        ), y se eleva a una potencia, que es igual al peso (      ) de cada criterio. Es decir criterio ^ peso_criterio.
MMI_t12 = pandas.DataFrame(MMI_mar.copy())
for columna in MMI_t12.columns.tolist():
    for fila in MMI_t12.index:
        MMI_t12[columna][fila] = numpy.power( MMI_t12[columna][fila], ndp[columna])
#print(MMI_t12)

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se multipliquen y los de costo se dividan. Igual que en el paso anterior, se toma cada elemento de esa nueva Matriz generada y se eleva a la potencia que le corresponde a su criterio.
MMI_t13 = pandas.DataFrame(MMI_t12.copy())
for columna in MMI_t13.columns.tolist():
    for fila in MMI_t13.index:
        MMI_t13[columna][fila] = numpy.power( MMI_t13[columna][fila], vectorBenCos[columna])
#print(MMI_t13)

#Para el tercer ranking (MMI_FMF), se utiliza nueva la matrix generada, y se multiplican todos los criterios de  cada actividad. Ejemplo Evento1-Actividad1 : criterio1*criterio2*criterio3. . .criterioN. El índice del ranking es el orden de descendente del resultado
MMI_FMF = {}
for fila in actividad_indices:
    MMI_FMF[fila] = MMI_t13.loc[fila].product()
MMI_FMF = pandas.Series(MMI_FMF.copy())
MMI_FMF = MMI_FMF.to_frame(name='u_i')
MMI_FMF['ranking'] = MMI_FMF.rank(ascending = False)
#print(MMI_FMF)

#Para utilizar los métodos MMI_RPM e MMI_IMB y obtener el ranking final de los eventos, necesitaremos los índices y con sus respectivos rankings de cada actividad, obtenidos de los tres métodos anteriores.
#Matriz con los rankings numéricos
MMI_rn = pandas.DataFrame()
MMI_rn['y_i'] = MMI_RS['ranking']
MMI_rn['z_i'] = MMI_i_RPA['ranking']
MMI_rn['u_i'] = MMI_FMF['ranking']
#print(MMI_rn)

#Matriz con los los índices
MMI_mi = pandas.DataFrame()
MMI_mi['y_i'] = MMI_RS['y_i']
MMI_mi['z_i'] = MMI_i_RPA['z_i']
MMI_mi['u_i'] = MMI_FMF['u_i']
#print(MMI_mi)
#Se toman los índices y se suman sus cuadrados
MMI_smi2 = {}
MMI_smi2['y_i'] = (MMI_mi['y_i'] ** 2).sum()
MMI_smi2['z_i'] = (MMI_mi['z_i'] ** 2).sum()
MMI_smi2['u_i'] = (MMI_mi['u_i'] ** 2).sum()
MMI_smi2 = pandas.Series(MMI_smi2)
#print(MMI_smi2)

#Indices normalizados
MMI_IN = pandas.DataFrame(MMI_mi.copy())
for columna in MMI_IN.columns.tolist():
    for fila in MMI_IN.index:
        MMI_IN[columna][fila] = MMI_IN[columna][fila] / MMI_smi2[columna]
#print(MMI_IN)

#MMI_RPM(A1)
MMI_RMP = pandas.DataFrame()
MMI_RMP['MMI_RMP'] = MMI_mi['y_i']
for fila in MMI_RMP.index:
    suma = 0
    for columna in MMI_rn:
        suma += 1 / MMI_rn[columna][fila]
    MMI_RMP['MMI_RMP'][fila] = 1 / suma
MMI_RMP['ranking'] = MMI_RMP.rank()
#print(MMI_RMP)

MMI_RPM = pandas.DataFrame(MMI_RMP.copy())
MMI_RPM = MMI_RPM.sort_values(by=['ranking'])
#print(MMI_RPM)

#MMI_m son los elementos que se van a rankear  MMI_m =
MMI_m = MMI_RPM.shape[0]
#print(MMI_m)
#MMI_m(MMI_m+1)/2
MMI_mm = MMI_m * ( MMI_m + 1) / 2
#print(MMI_mm)

MMI_IMB = pandas.DataFrame()
MMI_IMB['MMI_IMB'] = MMI_mi['y_i']
for fila in MMI_IMB.index:
    MMI_IMB['MMI_IMB'][fila] = (MMI_IN['y_i'][fila] * ((MMI_m - MMI_rn['y_i'][fila] + 1)/MMI_mm)) - (MMI_IN['z_i'][fila] * (MMI_rn['z_i'][fila]/MMI_mm)) + (MMI_IN['u_i'][fila] * ((MMI_m - MMI_rn['u_i'][fila] + 1)/MMI_mm))
MMI_IMB['ranking'] = MMI_IMB.rank(ascending = False)
#print(MMI_IMB)

MMI_IMBo = pandas.DataFrame(MMI_IMB.copy())
MMI_IMBo = MMI_IMBo.sort_values(by=['ranking'])
#print(MMI_IMBo)










#multimoora superior
#Se calculan las medias aritméticas de cada uno de los criterios( x ), usando el DataSet con los valores inferiores normalizados
filas = numpy.zeros(len(actividad_indices))
columnas = {}
for columna in dsSuperior.columns.tolist():
    columnas[columna] = filas

MMS_ma = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsSuperior.columns.tolist():
    for fila in actividad_indices:
        MMS_ma[columna][fila] = dsSuperior.iloc[actividad_indices[fila]][columna].mean()
#print(MMS_ma)

#Se elevan al cuadrado cada uno de los elementos (x^2) de la matriz anterior
MMS_ma2 = MMS_ma.applymap(lambda x: x * x)

#Se suman los cuadrados de cada criterio, y obtenemos un array que usaremos a continuación
MMS_suma_ma = {}
for columna in MMS_ma2:
    MMS_suma_ma[columna] = MMS_ma2[columna].sum()

#Se genera una  Matriz ()  tomando de cada criterio la media aritmética por cada evento y se divide entre la raiz cuadrada de la suma de los cuadrados () de cada criterio
MMS_mar = pandas.DataFrame(MMS_ma.copy())
#print(MMS_mar)
for columna in MMS_mar.columns.tolist():
    for fila in actividad_indices:
        MMS_mar[columna][fila] = MMS_mar[columna][fila] / numpy.sqrt( MMS_suma_ma[columna] )

#Se multiplica cada elemento de cada criterio de la Matriz generada, por el peso asociado a ese criterio utilizando el vector de Normalización de pesos. La nueva Matriz es (             )
MMS_marm = pandas.DataFrame(MMS_mar.copy())
for columna in MMS_marm.columns.tolist():
    for fila in actividad_indices:
        MMS_marm[columna][fila] = MMS_mar[columna][fila] * ndp[columna]

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se sumen y los de costo se resten
MMS_marm_v_bc = pandas.DataFrame(MMS_marm.copy())
for columna in MMS_marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        MMS_marm_v_bc[columna][fila] = MMS_marm_v_bc[columna][fila] * vectorBenCos[columna]
#print(MMS_marm_v_bc)

#Para obtener el primer ranking (MMS_RS), se suman los criterios de beneficio o restan los criterios de costo por evento. El índice del ranking es del valor mayor al menor
MMS_RS = {}
for fila in actividad_indices:
    MMS_RS[fila] = MMS_marm_v_bc.loc[fila].sum()
MMS_RS = pandas.Series(MMS_RS.copy())
MMS_RS = MMS_RS.to_frame(name='y_i')
MMS_RS['ranking'] = MMS_RS.rank(ascending = False)
#print(MMS_RS)

#Para el siguiente ranking, se generan dos arrays, uno con los máximos y otros con los mínimos de los criterios en la Matriz (       ), los máximos representan los criterios de beneficios y los mínimos los criterios de costo.
MMS_t9 = pandas.DataFrame()
MMS_t9 = MMS_t9.append( pandas.Series(MMS_mar.max(), name='max') )
MMS_t9 = MMS_t9.append( pandas.Series(MMS_mar.min(), name='min') )
#print(MMS_t9)

#Para calcular el array de referencia, tomaremos los valores máximos de cada criterio que son beneficio y los mínimos de costo, ayudandonos del Vector Ben(+)/Cost(-), y los multiplicaremos por el peso (    ) que cada criterio tiene.
MMS_w_r_r_j = {}
for fila in vectorBenCos.index:
    MMS_w_r_r_j[fila] = MMS_t9[fila]['max'] * ndp[fila] if vectorBenCos[fila] > 0 else MMS_t9[fila]['min'] * ndp[fila]
MMS_w_r_r_j = pandas.Series(MMS_w_r_r_j.copy())
#print(MMS_w_r_r_j)

#Para calcular el segundo ranking (MMS_RPA), a cada valor del criterio en la matriz (           ), se le resta el valor de ese criterio en el array de referencia (          ).
MMS_RPA = pandas.DataFrame(MMS_marm.copy())
for columna in MMS_RPA.columns.tolist():
    for fila in MMS_RPA.index:
        if MMS_RPA[columna][fila] != numpy.nan:
            MMS_RPA[columna][fila] = MMS_w_r_r_j[columna] - MMS_RPA[columna][fila]
#print(MMS_RPA)

#Para obtener el segundo ranking (MMS_RPA), se toman el valor máximo de cada evento. El índice del ranking es el órden ascendente de los valores anteriores
MMS_i_RPA = {}
for fila in actividad_indices:
    MMS_i_RPA[fila] = MMS_RPA.loc[fila].max()
MMS_i_RPA = pandas.Series(MMS_i_RPA.copy())
MMS_i_RPA = MMS_i_RPA.to_frame(name='z_i')
MMS_i_RPA['ranking'] = MMS_i_RPA.rank()
#print(MMS_i_RPA)

#Para el último ranking, se toma cada criterio de la matriz (        ), y se eleva a una potencia, que es igual al peso (      ) de cada criterio. Es decir criterio ^ peso_criterio.
MMS_t12 = pandas.DataFrame(MMS_mar.copy())
for columna in MMS_t12.columns.tolist():
    for fila in MMS_t12.index:
        MMS_t12[columna][fila] = numpy.power( MMS_t12[columna][fila], ndp[columna])
#print(MMS_t12)

#Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se multipliquen y los de costo se dividan. Igual que en el paso anterior, se toma cada elemento de esa nueva Matriz generada y se eleva a la potencia que le corresponde a su criterio.
MMS_t13 = pandas.DataFrame(MMS_t12.copy())
for columna in MMS_t13.columns.tolist():
    for fila in MMS_t13.index:
        MMS_t13[columna][fila] = numpy.power( MMS_t13[columna][fila], vectorBenCos[columna])
#print(MMS_t13)

#Para el tercer ranking (MMS_FMF), se utiliza nueva la matrix generada, y se multiplican todos los criterios de  cada actividad. Ejemplo Evento1-Actividad1 : criterio1*criterio2*criterio3. . .criterioN. El índice del ranking es el orden de descendente del resultado
MMS_FMF = {}
for fila in actividad_indices:
    MMS_FMF[fila] = MMS_t13.loc[fila].product()
MMS_FMF = pandas.Series(MMS_FMF.copy())
MMS_FMF = MMS_FMF.to_frame(name='u_i')
MMS_FMF['ranking'] = MMS_FMF.rank(ascending = False)
#print(MMS_FMF)

#Para utilizar los métodos MMS_RPM e MMS_IMB y obtener el ranking final de los eventos, necesitaremos los índices y con sus respectivos rankings de cada actividad, obtenidos de los tres métodos anteriores.
#Matriz con los rankings numéricos
MMS_rn = pandas.DataFrame()
MMS_rn['y_i'] = MMS_RS['ranking']
MMS_rn['z_i'] = MMS_i_RPA['ranking']
MMS_rn['u_i'] = MMS_FMF['ranking']
#print(MMS_rn)

#Matriz con los los índices
MMS_mi = pandas.DataFrame()
MMS_mi['y_i'] = MMS_RS['y_i']
MMS_mi['z_i'] = MMS_i_RPA['z_i']
MMS_mi['u_i'] = MMS_FMF['u_i']
#print(MMS_mi)
#Se toman los índices y se suman sus cuadrados
MMS_smi2 = {}
MMS_smi2['y_i'] = (MMS_mi['y_i'] ** 2).sum()
MMS_smi2['z_i'] = (MMS_mi['z_i'] ** 2).sum()
MMS_smi2['u_i'] = (MMS_mi['u_i'] ** 2).sum()
MMS_smi2 = pandas.Series(MMS_smi2)
#print(MMS_smi2)

#Indices normalizados
MMS_IN = pandas.DataFrame(MMS_mi.copy())
for columna in MMS_IN.columns.tolist():
    for fila in MMS_IN.index:
        MMS_IN[columna][fila] = MMS_IN[columna][fila] / MMS_smi2[columna]
#print(MMS_IN)

#MMS_RPM(A1)
MMS_RMP = pandas.DataFrame()
MMS_RMP['MMS_RMP'] = MMS_mi['y_i']
for fila in MMS_RMP.index:
    suma = 0
    for columna in MMS_rn:
        suma += 1 / MMS_rn[columna][fila]
    MMS_RMP['MMS_RMP'][fila] = 1 / suma
MMS_RMP['ranking'] = MMS_RMP.rank()
#print(MMS_RMP)

MMS_RPM = pandas.DataFrame(MMS_RMP.copy())
MMS_RPM = MMS_RPM.sort_values(by=['ranking'])
#print(MMS_RPM)

#MMS_m son los elementos que se van a rankear  MMS_m =
MMS_m = MMS_RPM.shape[0]
#print(MMS_m)
#MMS_m(MMS_m+1)/2
MMS_mm = MMS_m * ( MMS_m + 1) / 2
#print(MMS_mm)

MMS_IMB = pandas.DataFrame()
MMS_IMB['MMS_IMB'] = MMS_mi['y_i']
for fila in MMS_IMB.index:
    MMS_IMB['MMS_IMB'][fila] = (MMS_IN['y_i'][fila] * ((MMS_m - MMS_rn['y_i'][fila] + 1)/MMS_mm)) - (MMS_IN['z_i'][fila] * (MMS_rn['z_i'][fila]/MMS_mm)) + (MMS_IN['u_i'][fila] * ((MMS_m - MMS_rn['u_i'][fila] + 1)/MMS_mm))
MMS_IMB['ranking'] = MMS_IMB.rank(ascending = False)
#print(MMS_IMB)

MMS_IMBo = pandas.DataFrame(MMS_IMB.copy())
MMS_IMBo = MMS_IMBo.sort_values(by=['ranking'])
print(MMS_IMBo)
