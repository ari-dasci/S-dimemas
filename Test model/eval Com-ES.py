#!/bin/python
# -*- coding: utf-8 -*-

import math
import pandas
import re
import numpy
import os
import json
import shutil
import sys
from statistics import mean
from decimal import Decimal, ROUND_HALF_UP

nround = lambda number: int(Decimal(number).to_integral_value(rounding=ROUND_HALF_UP))

if len(sys.argv) != 6:
    print("Número de argumentos inválido")
    sys.exit(1)
rho = float(sys.argv[4])
# rho = 0.5
lang = sys.argv[5]
# lang = 'es'
cfgFile = sys.argv[2]
# cfgFile = 'info.js'
outputDir = sys.argv[3]
# outputDir = 'proyectos/prueba/'
rawData = pandas.read_csv(sys.argv[1])
# rawData = pandas.read_csv('datos-lanoche.csv')

# carga de los datos
# se cargan todos los datos del origen
# se crea una lista solo con las escalas
escala = pandas.Series(rawData['ESCALA'])
# se determina cual es la columna de las escalas para poder extraer a posterior solamente las evaluaciones de los usuarios
cEscala = rawData.columns.get_loc('ESCALA')
datos = rawData.iloc[:, cEscala+1:]  # se extraen unicamente las evaluaciones
tactividades = rawData.ACTIVIDAD
# se determinan las diferentes actividades por titulo
actividades = rawData.ACTIVIDAD.unique()
actividad_indices = {}
for actividad in actividades:
    # se obtienen los indices de las escalas correpondientes por actividad
    indices_escalas = rawData.index[rawData['ACTIVIDAD'] == actividad].tolist()
    actividad_indices[actividad] = indices_escalas

# obtención de los dataset por medio de expresiones regulares
# na_action no podía ser aplicado por la versión de pandas
dsPrioridades = datos.map(lambda x: int(
    re.search('(\d+)[+-]', x)[1]), na_action='ignore')
dsEvaluacion = datos.map(lambda x: re.search('[+-]', x)[0], na_action='ignore')
dsInferior = datos.map(lambda x: int(
    re.search('(\d+):', x)[1]), na_action='ignore')
dsSuperior = datos.map(lambda x: int(
    re.search(':(\d+)', x)[1]), na_action='ignore')

# print(dsInferior)
# print(dsSuperior)

# normalización

MCM = numpy.lcm.reduce(list(map(lambda x: x-1, rawData.ESCALA.unique())))
for i in dsInferior.index:
    nactividad = rawData.ACTIVIDAD.loc[i]
    Es = escala[i]
    for j in dsInferior:
        Ev = dsInferior[j][i]
        dsInferior[j][i] = ((Ev-1)*MCM/(Es-1))+1

        Ev = dsSuperior[j][i]
        dsSuperior[j][i] = ((Ev-1)*MCM/(Es-1))+1
# print(dsInferior)
# print(dsSuperior)

# Pesos y vector costo - beneficio
"""
Se va a trabajar con diferentes pesos, éstos pesos se aplicarán para el modelo de evaluación y para el concenso opinion de cada evento. 
El primer peso se calculará  el pesos por elección de criterio que va de 1  a n, siendo 1 el primero en ser elegido y n el último. Sacamos el promedio de cada uno y despues se hace un reparto inversamente
"""
# Promedio de cada criterio
promedioscriterios = {}
for columna in dsPrioridades:
    promedioscriterios[columna] = dsPrioridades[columna].mean()
promedioscriterios = pandas.Series(promedioscriterios.copy())  # resultado

"""
Para saber el peso de cada criterio, se utiliza el reparto inversamente proporcional, que dará mayor peso a el promedio mas cercano a 1 y el menor peso al promedio mas cercano a n. Ejemplo: MAYOR PESO → 1. menor peso → 9
"""
MCM = numpy.lcm.reduce([int(valor) for valor in promedioscriterios.tolist()])
pesoscriterios = pandas.Series(promedioscriterios.copy())
pesoscriterios = pesoscriterios.apply(lambda x: (MCM / x) * 1)
suma = pesoscriterios.sum()
pesoscriterios = pesoscriterios.apply(lambda x: (1 * x) / suma)  # resultado
sumadepesos = pesoscriterios.sum()

"""
El segundo peso que se va a calcular será a partir del porcentaje de participación. Es decir, por el número de veces que un criterio fue evaluado con respecto al total. A mayor evaluaciones en un criterio, mayor peso
Para calcular éste peso, se puede hacer el coteo de los criterios en cualquiera de los DataSets que se generaron de los datos de entrada (rawDATA)
"""
pesoparticipacion = {}
for columna in dsPrioridades:
    fdatos = pandas.Series(dsPrioridades[columna].copy())
    pesoparticipacion[columna] = fdatos[fdatos > 0].count()
pesoparticipacion = pandas.Series(pesoparticipacion.copy())
suma = pesoparticipacion.sum()
pesoparticipacion = pesoparticipacion.apply(lambda x: x / suma)  # resultado
sumadepesos = pesoparticipacion.sum()

vectorBenCos = {}
for columna in dsEvaluacion:
    fdatos = pandas.Series(dsEvaluacion[columna].copy())
    vectorBenCos[columna] = 1 if fdatos[fdatos ==
                                        '+'].count() >= fdatos[fdatos == '-'].count() else -1
vectorBenCos = pandas.Series(vectorBenCos.copy())  # resultado


# por dimensiones
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
    pesosdimensiones[dimension] = numpy.nanmean(fdatos)
pesosdimensiones = pandas.Series(pesosdimensiones.copy())

MCM = numpy.lcm.reduce([int(x) for x in pesosdimensiones.tolist()])
pesosdimensiones = pesosdimensiones.apply(lambda x: (MCM / x) * 1)
suma = pesosdimensiones.sum()
pesosdimensiones = pesosdimensiones.apply(
    lambda x: (1 * x) / suma)  # resultado
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
ppdimension = ppdimension.apply(lambda x: x / suma)  # resultado
sumadepesos = ppdimension.sum()


# multimoora inferior
# Utilizando el vector de pesos por importancia y el vector de pesos por participación, se calcula el peso final que tendrá cada criterio
ndp = pesoscriterios.mul(pesoparticipacion)
suma = ndp.sum()
ndp = ndp.apply(lambda x: x / suma)  # resultado de Normalización de pesos

# Se calculan las medias aritméticas de cada uno de los criterios( x ), usando el DataSet con los valores inferiores normalizados
filas = numpy.zeros(len(actividad_indices))
columnas = {}
for columna in dsInferior.columns.tolist():
    columnas[columna] = filas

MMI_ma = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsInferior.columns.tolist():
    for fila in actividad_indices:
        MMI_ma[columna][fila] = dsInferior.iloc[actividad_indices[fila]
                                                ][columna].mean()
# print(MMI_ma)

# Se elevan al cuadrado cada uno de los elementos (x^2) de la matriz anterior
MMI_ma2 = MMI_ma.map(lambda x: x * x)

# Se suman los cuadrados de cada criterio, y obtenemos un array que usaremos a continuación
MMI_suma_ma = {}
for columna in MMI_ma2:
    MMI_suma_ma[columna] = MMI_ma2[columna].sum()

# Se genera una  Matriz ()  tomando de cada criterio la media aritmética por cada evento y se divide entre la raiz cuadrada de la suma de los cuadrados () de cada criterio
MMI_mar = pandas.DataFrame(MMI_ma.copy())
# print(MMI_mar)
for columna in MMI_mar.columns.tolist():
    for fila in actividad_indices:
        MMI_mar[columna][fila] = MMI_mar[columna][fila] / \
            numpy.sqrt(MMI_suma_ma[columna])

# Se multiplica cada elemento de cada criterio de la Matriz generada, por el peso asociado a ese criterio utilizando el vector de Normalización de pesos. La nueva Matriz es (             )
MMI_marm = pandas.DataFrame(MMI_mar.copy())
for columna in MMI_marm.columns.tolist():
    for fila in actividad_indices:
        MMI_marm[columna][fila] = MMI_mar[columna][fila] * ndp[columna]

# Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se sumen y los de costo se resten
MMI_marm_v_bc = pandas.DataFrame(MMI_marm.copy())
for columna in MMI_marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        MMI_marm_v_bc[columna][fila] = MMI_marm_v_bc[columna][fila] * \
            vectorBenCos[columna]
# print(MMI_marm_v_bc)

# Para obtener el primer ranking (MMI_RS), se suman los criterios de beneficio o restan los criterios de costo por evento. El índice del ranking es del valor mayor al menor
MMI_RS = {}
for fila in actividad_indices:
    MMI_RS[fila] = MMI_marm_v_bc.loc[fila].sum()
MMI_RS = pandas.Series(MMI_RS.copy())
MMI_RS = MMI_RS.to_frame(name='y_i')
MMI_RS['ranking'] = MMI_RS.rank(ascending=False)
# print(MMI_RS)

# Para el siguiente ranking, se generan dos arrays, uno con los máximos y otros con los mínimos de los criterios en la Matriz (       ), los máximos representan los criterios de beneficios y los mínimos los criterios de costo.
MMI_t9 = pandas.DataFrame()
MMI_t9['max'] = MMI_mar.max()
MMI_t9['min'] = MMI_mar.min()
# print(MMI_t9)

# Para calcular el array de referencia, tomaremos los valores máximos de cada criterio que son beneficio y los mínimos de costo, ayudandonos del Vector Ben(+)/Cost(-), y los multiplicaremos por el peso (    ) que cada criterio tiene.
MMI_w_r_r_j = {}
for fila in vectorBenCos.index:
    MMI_w_r_r_j[fila] = MMI_t9['max'][fila] * \
        ndp[fila] if vectorBenCos[fila] > 0 else MMI_t9['min'][fila] * ndp[fila]
MMI_w_r_r_j = pandas.Series(MMI_w_r_r_j.copy())
# print(MMI_w_r_r_j)

# Para calcular el segundo ranking (MMI_RPA), a cada valor del criterio en la matriz (           ), se le resta el valor de ese criterio en el array de referencia (          ).
MMI_RPA = pandas.DataFrame(MMI_marm.copy())
for columna in MMI_RPA.columns.tolist():
    for fila in MMI_RPA.index:
        if MMI_RPA[columna][fila] != numpy.nan:
            MMI_RPA[columna][fila] = abs(
                MMI_w_r_r_j[columna] - MMI_RPA[columna][fila])
# print(MMI_RPA)

# Para obtener el segundo ranking (MMI_RPA), se toman el valor máximo de cada evento. El índice del ranking es el órden ascendente de los valores anteriores
MMI_i_RPA = {}
for fila in actividad_indices:
    MMI_i_RPA[fila] = MMI_RPA.loc[fila].max()
MMI_i_RPA = pandas.Series(MMI_i_RPA.copy())
MMI_i_RPA = MMI_i_RPA.to_frame(name='z_i')
MMI_i_RPA['ranking'] = MMI_i_RPA.rank()
# print(MMI_i_RPA)

# Para el último ranking, se toma cada criterio de la matriz (        ), y se eleva a una potencia, que es igual al peso (      ) de cada criterio. Es decir criterio ^ peso_criterio.
MMI_t12 = pandas.DataFrame(MMI_mar.copy())
for columna in MMI_t12.columns.tolist():
    for fila in MMI_t12.index:
        MMI_t12[columna][fila] = numpy.power(
            MMI_t12[columna][fila], ndp[columna])
# print(MMI_t12)

# Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se multipliquen y los de costo se dividan. Igual que en el paso anterior, se toma cada elemento de esa nueva Matriz generada y se eleva a la potencia que le corresponde a su criterio.
MMI_t13 = pandas.DataFrame(MMI_t12.copy())
for columna in MMI_t13.columns.tolist():
    for fila in MMI_t13.index:
        MMI_t13[columna][fila] = numpy.power(
            MMI_t13[columna][fila], vectorBenCos[columna])
# print(MMI_t13)

# Para el tercer ranking (MMI_FMF), se utiliza nueva la matrix generada, y se multiplican todos los criterios de  cada actividad. Ejemplo Evento1-Actividad1 : criterio1*criterio2*criterio3. . .criterioN. El índice del ranking es el orden de descendente del resultado
MMI_FMF = {}
for fila in actividad_indices:
    MMI_FMF[fila] = MMI_t13.loc[fila].product()
MMI_FMF = pandas.Series(MMI_FMF.copy())
MMI_FMF = MMI_FMF.to_frame(name='u_i')
MMI_FMF['ranking'] = MMI_FMF.rank(ascending=False)
# print(MMI_FMF)

# Para utilizar los métodos MMI_RPM e MMI_IMB y obtener el ranking final de los eventos, necesitaremos los índices y con sus respectivos rankings de cada actividad, obtenidos de los tres métodos anteriores.
# Matriz con los rankings numéricos
MMI_rn = pandas.DataFrame()
MMI_rn['y_i'] = MMI_RS['ranking']
MMI_rn['z_i'] = MMI_i_RPA['ranking']
MMI_rn['u_i'] = MMI_FMF['ranking']
# print(MMI_rn)

# Matriz con los los índices
MMI_mi = pandas.DataFrame()
MMI_mi['y_i'] = MMI_RS['y_i']
MMI_mi['z_i'] = MMI_i_RPA['z_i']
MMI_mi['u_i'] = MMI_FMF['u_i']
# print(MMI_mi)
# Se toman los índices y se suman sus cuadrados
MMI_smi2 = {}
MMI_smi2['y_i'] = (MMI_mi['y_i'] ** 2).sum()
MMI_smi2['z_i'] = (MMI_mi['z_i'] ** 2).sum()
MMI_smi2['u_i'] = (MMI_mi['u_i'] ** 2).sum()
MMI_smi2 = pandas.Series(MMI_smi2)
# print(MMI_smi2)

# Indices normalizados
MMI_IN = pandas.DataFrame(MMI_mi.copy())
for columna in MMI_IN.columns.tolist():
    for fila in MMI_IN.index:
        MMI_IN[columna][fila] = MMI_IN[columna][fila] / \
            numpy.sqrt(MMI_smi2[columna])
# print("ïndices normalizados\n", MMI_IN)

# MMI_RPM(A1)
MMI_RMP = pandas.DataFrame()
MMI_RMP['MMI_RMP'] = MMI_mi['y_i']
for fila in MMI_RMP.index:
    suma = 0
    for columna in MMI_rn:
        suma += 1 / MMI_rn[columna][fila]
    MMI_RMP['MMI_RMP'][fila] = 1 / suma
MMI_RMP['ranking'] = MMI_RMP.rank()
# print(MMI_RMP)

MMI_RPM = pandas.DataFrame(MMI_RMP.copy())
MMI_RPM = MMI_RPM.sort_values(by=['ranking'])
# print(MMI_RPM)

# MMI_m son los elementos que se van a rankear  MMI_m =
MMI_m = MMI_RPM.shape[0]
# print(MMI_m)
# MMI_m(MMI_m+1)/2
MMI_mm = MMI_m * (MMI_m + 1) / 2
# print(MMI_mm)

MMI_IMB = pandas.DataFrame()
MMI_IMB['MMI_IMB'] = MMI_mi['y_i']
for fila in MMI_IMB.index:
    MMI_IMB['MMI_IMB'][fila] = (MMI_IN['y_i'][fila] * ((MMI_m - MMI_rn['y_i'][fila] + 1)/MMI_mm)) - (
        MMI_IN['z_i'][fila] * (MMI_rn['z_i'][fila]/MMI_mm)) + (MMI_IN['u_i'][fila] * ((MMI_m - MMI_rn['u_i'][fila] + 1)/MMI_mm))
MMI_IMB['ranking'] = MMI_IMB.rank(ascending=False)
# print(MMI_IMB)

MMI_IMBo = pandas.DataFrame(MMI_IMB.copy())
MMI_IMBo = MMI_IMBo.sort_values(by=['ranking'])
# print(MMI_IMBo)


# multimoora superior
# Se calculan las medias aritméticas de cada uno de los criterios( x ), usando el DataSet con los valores inferiores normalizados
filas = numpy.zeros(len(actividad_indices))
columnas = {}
for columna in dsSuperior.columns.tolist():
    columnas[columna] = filas

MMS_ma = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsSuperior.columns.tolist():
    for fila in actividad_indices:
        MMS_ma[columna][fila] = dsSuperior.iloc[actividad_indices[fila]
                                                ][columna].mean()
# print(MMS_ma)

# Se elevan al cuadrado cada uno de los elementos (x^2) de la matriz anterior
MMS_ma2 = MMS_ma.map(lambda x: x * x)

# Se suman los cuadrados de cada criterio, y obtenemos un array que usaremos a continuación
MMS_suma_ma = {}
for columna in MMS_ma2:
    MMS_suma_ma[columna] = MMS_ma2[columna].sum()

# Se genera una  Matriz ()  tomando de cada criterio la media aritmética por cada evento y se divide entre la raiz cuadrada de la suma de los cuadrados () de cada criterio
MMS_mar = pandas.DataFrame(MMS_ma.copy())
# print(MMS_mar)
for columna in MMS_mar.columns.tolist():
    for fila in actividad_indices:
        MMS_mar[columna][fila] = MMS_mar[columna][fila] / \
            numpy.sqrt(MMS_suma_ma[columna])

# Se multiplica cada elemento de cada criterio de la Matriz generada, por el peso asociado a ese criterio utilizando el vector de Normalización de pesos. La nueva Matriz es (             )
MMS_marm = pandas.DataFrame(MMS_mar.copy())
for columna in MMS_marm.columns.tolist():
    for fila in actividad_indices:
        MMS_marm[columna][fila] = MMS_mar[columna][fila] * ndp[columna]

# Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se sumen y los de costo se resten
MMS_marm_v_bc = pandas.DataFrame(MMS_marm.copy())
for columna in MMS_marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        MMS_marm_v_bc[columna][fila] = MMS_marm_v_bc[columna][fila] * \
            vectorBenCos[columna]
# print(MMS_marm_v_bc)

# Para obtener el primer ranking (MMS_RS), se suman los criterios de beneficio o restan los criterios de costo por evento. El índice del ranking es del valor mayor al menor
MMS_RS = {}
for fila in actividad_indices:
    MMS_RS[fila] = MMS_marm_v_bc.loc[fila].sum()
MMS_RS = pandas.Series(MMS_RS.copy())
MMS_RS = MMS_RS.to_frame(name='y_i')
MMS_RS['ranking'] = MMS_RS.rank(ascending=False)
print(MMS_RS)

# Para el siguiente ranking, se generan dos arrays, uno con los máximos y otros con los mínimos de los criterios en la Matriz (       ), los máximos representan los criterios de beneficios y los mínimos los criterios de costo.
MMS_t9 = pandas.DataFrame()
MMS_t9['max'] = MMS_mar.max()
MMS_t9['min'] = MMS_mar.min()
# print(MMS_t9)

# Para calcular el array de referencia, tomaremos los valores máximos de cada criterio que son beneficio y los mínimos de costo, ayudandonos del Vector Ben(+)/Cost(-), y los multiplicaremos por el peso (    ) que cada criterio tiene.
MMS_w_r_r_j = {}
for fila in vectorBenCos.index:
    MMS_w_r_r_j[fila] = MMS_t9['max'][fila] * \
        ndp[fila] if vectorBenCos[fila] > 0 else MMS_t9['min'][fila] * ndp[fila]
MMS_w_r_r_j = pandas.Series(MMS_w_r_r_j.copy())
# print(MMS_w_r_r_j)

# Para calcular el segundo ranking (MMS_RPA), a cada valor del criterio en la matriz (           ), se le resta el valor de ese criterio en el array de referencia (          ).
MMS_RPA = pandas.DataFrame(MMS_marm.copy())
for columna in MMS_RPA.columns.tolist():
    for fila in MMS_RPA.index:
        if MMS_RPA[columna][fila] != numpy.nan:
            MMS_RPA[columna][fila] = abs(
                MMS_w_r_r_j[columna] - MMS_RPA[columna][fila])#Corregida, faltaba valor absoluto
print(MMS_RPA)

# Para obtener el segundo ranking (MMS_RPA), se toman el valor máximo de cada evento. El índice del ranking es el órden ascendente de los valores anteriores
MMS_i_RPA = {}
for fila in actividad_indices:
    MMS_i_RPA[fila] = MMS_RPA.loc[fila].max()
MMS_i_RPA = pandas.Series(MMS_i_RPA.copy())
MMS_i_RPA = MMS_i_RPA.to_frame(name='z_i')
MMS_i_RPA['ranking'] = MMS_i_RPA.rank()
# print(MMS_i_RPA)

# Para el último ranking, se toma cada criterio de la matriz (        ), y se eleva a una potencia, que es igual al peso (      ) de cada criterio. Es decir criterio ^ peso_criterio.
MMS_t12 = pandas.DataFrame(MMS_mar.copy())
for columna in MMS_t12.columns.tolist():
    for fila in MMS_t12.index:
        MMS_t12[columna][fila] = numpy.power(
            MMS_t12[columna][fila], ndp[columna])
# print(MMS_t12)

# Se utiliza el Vector Ben(+)/Cost(-) para que los criterios de beneficio se multipliquen y los de costo se dividan. Igual que en el paso anterior, se toma cada elemento de esa nueva Matriz generada y se eleva a la potencia que le corresponde a su criterio.
MMS_t13 = pandas.DataFrame(MMS_t12.copy())
for columna in MMS_t13.columns.tolist():
    for fila in MMS_t13.index:
        MMS_t13[columna][fila] = numpy.power(
            MMS_t13[columna][fila], vectorBenCos[columna])
# print(MMS_t13)

# Para el tercer ranking (MMS_FMF), se utiliza nueva la matrix generada, y se multiplican todos los criterios de  cada actividad. Ejemplo Evento1-Actividad1 : criterio1*criterio2*criterio3. . .criterioN. El índice del ranking es el orden de descendente del resultado
MMS_FMF = {}
for fila in actividad_indices:
    MMS_FMF[fila] = MMS_t13.loc[fila].product()
MMS_FMF = pandas.Series(MMS_FMF.copy())
MMS_FMF = MMS_FMF.to_frame(name='u_i')
MMS_FMF['ranking'] = MMS_FMF.rank(ascending=False)
print(MMS_FMF)

# Para utilizar los métodos MMS_RPM e MMS_IMB y obtener el ranking final de los eventos, necesitaremos los índices y con sus respectivos rankings de cada actividad, obtenidos de los tres métodos anteriores.
# Matriz con los rankings numéricos
MMS_rn = pandas.DataFrame()
MMS_rn['y_i'] = MMS_RS['ranking']
MMS_rn['z_i'] = MMS_i_RPA['ranking']
MMS_rn['u_i'] = MMS_FMF['ranking']
# print(MMS_rn)

# Matriz con los los índices
MMS_mi = pandas.DataFrame()
MMS_mi['y_i'] = MMS_RS['y_i']
MMS_mi['z_i'] = MMS_i_RPA['z_i']
MMS_mi['u_i'] = MMS_FMF['u_i']
# print(MMS_mi)
# Se toman los índices y se suman sus cuadrados
MMS_smi2 = {}
MMS_smi2['y_i'] = (MMS_mi['y_i'] ** 2).sum()
MMS_smi2['z_i'] = (MMS_mi['z_i'] ** 2).sum()
MMS_smi2['u_i'] = (MMS_mi['u_i'] ** 2).sum()
MMS_smi2 = pandas.Series(MMS_smi2)
# print(MMS_smi2)

# Indices normalizados
MMS_IN = pandas.DataFrame(MMS_mi.copy())
for columna in MMS_IN.columns.tolist():
    for fila in MMS_IN.index:
        MMS_IN[columna][fila] = MMS_IN[columna][fila] / numpy.sqrt(MMS_smi2[columna])#Corrección, faltaba raíz
# print(MMS_IN)

# MMS_RPM(A1)
MMS_RMP = pandas.DataFrame()
MMS_RMP['MMS_RMP'] = MMS_mi['y_i']
for fila in MMS_RMP.index:
    suma = 0
    for columna in MMS_rn:
        suma += 1 / MMS_rn[columna][fila]
    MMS_RMP['MMS_RMP'][fila] = 1 / suma
MMS_RMP['ranking'] = MMS_RMP.rank()
print(MMS_RMP)

MMS_RPM = pandas.DataFrame(MMS_RMP.copy())
MMS_RPM = MMS_RPM.sort_values(by=['ranking'])
print(MMS_RPM)

# MMS_m son los elementos que se van a rankear  MMS_m =
MMS_m = MMS_RPM.shape[0]
# print(MMS_m)
# MMS_m(MMS_m+1)/2
MMS_mm = MMS_m * (MMS_m + 1) / 2
# print(MMS_mm)

MMS_IMB = pandas.DataFrame()
MMS_IMB['MMS_IMB'] = MMS_mi['y_i']
for fila in MMS_IMB.index:
    MMS_IMB['MMS_IMB'][fila] = (MMS_IN['y_i'][fila] * ((MMS_m - MMS_rn['y_i'][fila] + 1)/MMS_mm)) - (
        MMS_IN['z_i'][fila] * (MMS_rn['z_i'][fila]/MMS_mm)) + (MMS_IN['u_i'][fila] * ((MMS_m - MMS_rn['u_i'][fila] + 1)/MMS_mm))
MMS_IMB['ranking'] = MMS_IMB.rank(ascending=False)
print(MMS_IMB)

MMS_IMBo = pandas.DataFrame(MMS_IMB.copy())
MMS_IMBo = MMS_IMBo.sort_values(by=['ranking'])
print(MMS_IMBo)


# Al trabajar con límites inferiores y superiores, podemos hacer un ranking tomando en cuenta ambos límites.
# Para este nuevo ranking, calcularemos la media aritmética de los índices superiores e inferiores de cada evento.
# índices inferiores
# print("índices inferiores")
MMH_ii_RS = pandas.DataFrame()
MMH_ii_RS['y_i'] = MMI_mi['y_i']
MMH_ii_RS['ranking'] = MMH_ii_RS.rank(ascending=False)

# print(MMH_ii_RS)

MMH_ii_RPA = pandas.DataFrame()
MMH_ii_RPA['z_i'] = MMI_mi['z_i']
MMH_ii_RPA['ranking'] = MMH_ii_RPA.rank(ascending=False)

# print(MMH_ii_RPA)

MMH_ii_FMF = pandas.DataFrame()
MMH_ii_FMF['u_i'] = MMI_mi['u_i']
MMH_ii_FMF['ranking'] = MMH_ii_FMF.rank(ascending=False)
# print(MMH_ii_FMF)

# indices superiores
# print("\n\níndices superiores")
MMH_is_RS = pandas.DataFrame()
MMH_is_RS['y_i'] = MMS_mi['y_i']
MMH_is_RS['ranking'] = MMH_is_RS.rank(ascending=False)

# print(MMH_is_RS)

MMH_is_RPA = pandas.DataFrame()
MMH_is_RPA['z_i'] = MMS_mi['z_i']
MMH_is_RPA['ranking'] = MMH_is_RPA.rank(ascending=False)

# print(MMH_is_RPA)

MMH_is_FMF = pandas.DataFrame()
MMH_is_FMF['u_i'] = MMS_mi['u_i']
MMH_is_FMF['ranking'] = MMH_is_FMF.rank(ascending=False)
# print(MMH_is_FMF)

# print("\n\nMedia de ambos índices")
MMH_mi_RS = pandas.DataFrame(MMH_ii_RS.copy())
MMH_mi_RPA = pandas.DataFrame(MMH_ii_RPA.copy())
MMH_mi_FMF = pandas.DataFrame(MMH_ii_FMF.copy())
for row in MMH_ii_RS.index:
    MMH_mi_RS['y_i'][row] = mean(
        [MMH_ii_RS['y_i'][row], MMH_is_RS['y_i'][row]])
    MMH_mi_RS['ranking'] = MMH_mi_RS['y_i'].rank(ascending=False)

    MMH_mi_RPA['z_i'][row] = mean(
        [MMH_ii_RPA['z_i'][row], MMH_is_RPA['z_i'][row]])
    MMH_mi_RPA['ranking'] = MMH_mi_RPA['z_i'].rank(ascending=True)

    MMH_mi_FMF['u_i'][row] = mean(
        [MMH_ii_FMF['u_i'][row], MMH_is_FMF['u_i'][row]])
    MMH_mi_FMF['ranking'] = MMH_mi_FMF['u_i'].rank(ascending=False)

# print(MMH_mi_RS)
# print(MMH_mi_RPA)
# print(MMH_mi_FMF)

# Para utilizar los métodos RPM e IMB y obtener el ranking final de los eventos, necesitaremos los índices y con sus respectivos rankings de cada actividad, obtenidos de RS, RPA y FMF
# Matriz con los rankings numéricos
# print("Matriz con los rankings numéricos")
MMH_rn = pandas.DataFrame()
MMH_rn['y_i'] = MMH_mi_RS['ranking']
MMH_rn['z_i'] = MMH_mi_RPA['ranking']
MMH_rn['u_i'] = MMH_mi_FMF['ranking']
# print(MMH_rn)

# Matriz con los los índices
MMH_mi = pandas.DataFrame()
MMH_mi['y_i'] = MMH_mi_RS['y_i']
MMH_mi['z_i'] = MMH_mi_RPA['z_i']
MMH_mi['u_i'] = MMH_mi_FMF['u_i']
# print(MMH_mi)
# Se toman los índices y se suman sus cuadrados
MMH_smi2 = {}
MMH_smi2['y_i'] = (MMH_mi['y_i'] ** 2).sum()
MMH_smi2['z_i'] = (MMH_mi['z_i'] ** 2).sum()
MMH_smi2['u_i'] = (MMH_mi['u_i'] ** 2).sum()
MMH_smi2 = pandas.Series(MMH_smi2)
# print(MMH_smi2)

# Indices normalizados
MMH_IN = pandas.DataFrame(MMH_mi.copy())
for columna in MMH_IN.columns.tolist():
    for fila in MMH_IN.index:
        MMH_IN[columna][fila] = MMH_IN[columna][fila] / MMH_smi2[columna]
# print(MMH_IN)

# MMH_RPM(A1)
MMH_RMP = pandas.DataFrame()
MMH_RMP['RMP'] = MMH_mi['y_i']
for fila in MMH_RMP.index:
    suma = 0
    for columna in MMH_rn:
        suma += 1 / MMH_rn[columna][fila]
    MMH_RMP['RMP'][fila] = 1 / suma
MMH_RMP['ranking'] = MMH_RMP.rank()
# print(MMS_RMP)

MMH_RPM = pandas.DataFrame(MMH_RMP.copy())
MMH_RPM = MMH_RPM.sort_values(by=['ranking'])
# print(MMH_RPM)

# MMS_m son los elementos que se van a rankear  MMS_m =
MMH_m = MMH_RPM.shape[0]
# print(MMH_m)
# MMS_m(MMS_m+1)/2
MMH_mm = MMH_m * (MMH_m + 1) / 2
# print(MMH_mm)


MMH_IMB = pandas.DataFrame()
MMH_IMB['IMB'] = MMH_mi['y_i']
for fila in MMH_IMB.index:
    MMH_IMB['IMB'][fila] = (MMH_IN['y_i'][fila] * ((MMH_m - MMH_rn['y_i'][fila] + 1)/MMH_mm)) - (MMH_IN['z_i']
                                                                                                 [fila] * (MMH_rn['z_i'][fila]/MMH_mm)) + (MMH_IN['u_i'][fila] * ((MMH_m - MMH_rn['u_i'][fila] + 1)/MMH_mm))
MMH_IMB['ranking'] = MMH_IMB.rank(ascending=False)
# print(MMH_IMB)

MMH_IMBo = pandas.DataFrame(MMH_IMB.copy())
MMH_IMBo = MMH_IMBo.sort_values(by=['ranking'])
# print(MMH_IMBo)

# Eval Ling por actividad
# Utilizando el vector de pesos por importancia y el vector de pesos por participación, se calcula el peso final que tendrá cada criterio
# Pesos por importancia = pesoscriterios
# Pesos por participación = pesoparticipacion
# w^s X w^o
Eval_Ling1 = {}
Eval_LingSum = 0
for columna in pesoscriterios.index:
    Eval_Ling1[columna] = pesoscriterios[columna] * pesoparticipacion[columna]
    Eval_LingSum += Eval_Ling1[columna]  # ∑(w^s X w^o)
# Ok
# print("\n", Eval_LingSum, "\n")
# Normalización de pesos
for columna in Eval_Ling1:
    Eval_Ling1[columna] = Eval_Ling1[columna] / Eval_LingSum
    suma += Eval_Ling1[columna]

Eval_Ling_t1 = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsInferior.columns.tolist():
    for fila in actividad_indices:
        Eval_Ling_t1[columna][fila] = dsInferior.iloc[actividad_indices[fila]][columna].mean()

Eval_Ling_Pesimista = pandas.DataFrame(Eval_Ling_t1.copy())
for columna in dsInferior.columns.tolist():
    for fila in actividad_indices:
        Eval_Ling_Pesimista[columna][fila] = Eval_Ling_Pesimista[columna][fila] * \
            Eval_Ling1[columna]


Elinguisticas = {
    'es': ['',
        'Pésimo',
        'Malo',
        'Muy pobre',
        'Pobre',
        'Suficiente',
        'Normal',
        'Bueno',
        'Muy bueno',
        'Bastante bueno',
        'Satisfecho	',
        'Muy Satisfecho',
        'Excelente',
        'Impresionante'
    ],'en': [
        '',
        'Terrible',
        'Nothing',
        'Very poor',
        'Poor',
        'Sufficient',
        'Average',
        'Good',
        'A Lot',
        'Quite a Lot',
        'Satisfied',
        'Very Satisfied',
        'Excellent',
        'Impressive'
    ]}

BetaAvg = []
S = []
Alpha = []
Etiquetas = []
for actividad in actividades:
    sumatoria = Eval_Ling_Pesimista.loc[actividad].sum()
    BetaAvg.append(sumatoria)
    S.append(nround(sumatoria))
    Alpha.append(sumatoria - nround(sumatoria))
    Etiquetas.append(Elinguisticas[lang][nround(sumatoria)])
Eval_Ling_Pesimista['BetaAvg'] = BetaAvg
Eval_Ling_Pesimista['S'] = S
Eval_Ling_Pesimista['Alpha'] = Alpha
Eval_Ling_Pesimista['Etiqueta_Linguistica'] = Etiquetas

# print(Eval_Ling_Pesimista)

Eval_Ling_t3 = pandas.DataFrame(columnas.copy(), actividad_indices.keys())
for columna in dsSuperior.columns.tolist():
    for fila in actividad_indices:
        Eval_Ling_t3[columna][fila] = dsSuperior.iloc[actividad_indices[fila]][columna].mean()

Eval_Ling_Optimista = pandas.DataFrame(Eval_Ling_t1.copy())
for columna in dsSuperior.columns.tolist():
    for fila in actividad_indices:
        Eval_Ling_Optimista[columna][fila] = Eval_Ling_t3[columna][fila] * \
            Eval_Ling1[columna]

BetaAvg = []
S = []
Alpha = []
Etiquetas = []
for actividad in actividades:
    sumatoria = Eval_Ling_Optimista.loc[actividad].sum()
    BetaAvg.append(sumatoria)
    S.append(nround(sumatoria))
    Alpha.append(sumatoria - nround(sumatoria))
    Etiquetas.append(Elinguisticas[lang][nround(sumatoria)])
Eval_Ling_Optimista['BetaAvg'] = BetaAvg
Eval_Ling_Optimista['S'] = S
Eval_Ling_Optimista['Alpha'] = Alpha
Eval_Ling_Optimista['Etiqueta_Linguistica'] = Etiquetas

Eval_Ling_Media = pandas.DataFrame(
    columns=['BetaAvg', 'S', 'Alpha', 'Etiqueta_Linguistica'])
for actividad in actividades:
    op = ((1 - rho) * Eval_Ling_Pesimista['BetaAvg'][actividad]
          ) + (rho * Eval_Ling_Optimista['BetaAvg'][actividad])
    Eval_Ling_Media.loc[actividad] = [
        op,
        nround(op),
        nround(op) - op,
        Elinguisticas[lang][nround(op)]
    ]
mean = Eval_Ling_Media['BetaAvg'].mean()
Evento_Eval_Ling_Media = {
    'BetaAvg': mean,
    'S': nround(mean),
    'Alpha': mean - nround(mean),
    'Etiqueta_Linguistica': Elinguisticas[lang][nround(mean)]
}

# Valoraciones por Dim(F)
VPD_Pesimista = pandas.DataFrame(columns=dimensiones)
tcolumnas = dsInferior.columns.tolist()
for actividad in actividades:
    datos = {}
    for dimension in dimensiones:
        columnas = [item for item in tcolumnas if item.startswith(dimension)]
        datos[dimension] = Eval_Ling_t1.loc[actividad, columnas].mean()
    VPD_Pesimista.loc[actividad] = datos
#print(VPD_Pesimista)

VPD_Optimista = pandas.DataFrame(columns=dimensiones)
tcolumnas = dsInferior.columns.tolist()
for actividad in actividades:
    datos = {}
    for dimension in dimensiones:
        columnas = [item for item in tcolumnas if item.startswith(dimension)]
        datos[dimension] = Eval_Ling_t3.loc[actividad, columnas].mean()
    VPD_Optimista.loc[actividad] = datos
#print(VPD_Optimista)

VPD_Evento = {
    'pesimista': {
        'BetaAvg': numpy.nanmean(VPD_Pesimista.values),
        'S': nround(numpy.nanmean(VPD_Pesimista.values)),
        'alpha': nround(numpy.nanmean(VPD_Pesimista.values)) - numpy.nanmean(VPD_Pesimista.values)
    },
    'optimista': {
        'BetaAvg': numpy.nanmean(VPD_Optimista.values),
        'S': nround(numpy.nanmean(VPD_Optimista.values)),
        'alpha': nround(numpy.nanmean(VPD_Optimista.values)) - numpy.nanmean(VPD_Optimista.values)
    }
}
VPD_Evento['media'] = {'BetaAvg': ((1 - rho) * VPD_Evento['pesimista']['BetaAvg'])+(rho * VPD_Evento['optimista']['BetaAvg'])}
VPD_Evento['media']['S'] = nround(VPD_Evento['media']['BetaAvg'])
VPD_Evento['media']['alpha'] = VPD_Evento['media']['S'] - VPD_Evento['media']['BetaAvg']
VPD_Evento['Etiqueta_Linguistica'] = Elinguisticas[lang][int(VPD_Evento['media']['S'])]
#print(VPD_Evento)

VPD = {}
for dimension in dimensiones:
    pesimista = pandas.DataFrame(columns=['BetaAvg', 'S', 'alpha'])
    optimista = pandas.DataFrame(columns=['BetaAvg', 'S', 'alpha'])
    media = pandas.DataFrame(columns=['BetaAvg', 'S', 'alpha', 'Etiqueta_Linguistica'])

    for actividad in actividades:
        VPDBetaAvgPesimista = VPD_Pesimista.loc[actividad, dimension]
        VPDBetaAvgOptimista = VPD_Optimista.loc[actividad, dimension]

        if math.isnan(VPDBetaAvgPesimista) or math.isnan(VPDBetaAvgOptimista):
            optimista.loc[actividad] = {'BetaAvg': '', 'S': '', 'alpha': ''}
            pesimista.loc[actividad] = {'BetaAvg': '', 'S': '', 'alpha': ''}
            media.loc[actividad] = {'BetaAvg': '', 'S': '', 'alpha': '', 'Etiqueta_Linguistica': ''}
        else:            
            VPDSPesimista = nround(VPDBetaAvgPesimista)
            VPDalphaPesimista = VPDBetaAvgPesimista - VPDSPesimista
            pesimista.loc[actividad] = {'BetaAvg': VPDBetaAvgPesimista, 'S': VPDSPesimista, 'alpha': VPDalphaPesimista}

            VPDSOptimista = nround(VPDBetaAvgOptimista)
            VPDalphaOptimista = VPDBetaAvgOptimista - VPDSOptimista
            optimista.loc[actividad] = {'BetaAvg': VPDBetaAvgOptimista, 'S': VPDSOptimista, 'alpha': VPDalphaOptimista}

            VPDBetaAvg = ((1-rho) * VPDBetaAvgPesimista) + (rho * VPDBetaAvgOptimista)
            
            VPDS = nround(VPDBetaAvg)
            VPDalpha = VPDBetaAvg - VPDS
            VPDetiqueta = Elinguisticas[lang][int(VPDS)]
            media.loc[actividad] = {'BetaAvg': VPDBetaAvg, 'S': VPDS, 'alpha': VPDalpha, 'Etiqueta_Linguistica': VPDetiqueta}

        VPD[dimension, 'pesimista'] = pesimista
        VPD[dimension, 'optimista'] = optimista
        VPD[dimension, 'media'] = media
#print(VPD)


shutil.rmtree(outputDir, ignore_errors=True)
shutil.copytree('base', outputDir)
shutil.copy(cfgFile, os.path.join(outputDir, 'datos', 'info.js'))

with open(os.path.join(outputDir, 'datos', 'config.js'), 'w') as f:
    f.write('var clang = "')
    f.write(lang)
    f.write('";')
    f.write('var rho = ')
    f.write(str(rho))
    f.write(';')

# Tablas informe
# print("Multimoora intervalo inferior")
dMMI_RS = MMI_RS.to_json(orient="index")
dMMI_RPA = MMI_i_RPA.to_json(orient="index")
dMMI_FMF = MMI_FMF.to_json(orient="index")
dMMI_RPM = MMI_RPM.to_json(orient="index")
dMMI_IMB = MMI_IMBo.to_json(orient="index")
with open(os.path.join(outputDir, 'datos', 'MMI.js'), 'w') as f:
    f.write('var MMI_RS = ')
    f.write(dMMI_RS)
    f.write(';')

    f.write('var MMI_RPA = ')
    f.write(dMMI_RPA)
    f.write(';')

    f.write('var MMI_FMF = ')
    f.write(dMMI_FMF)
    f.write(';')

    f.write('var MMI_RPM = ')
    f.write(dMMI_RPM)
    f.write(';')

    f.write('var MMI_IMB = ')
    f.write(dMMI_IMB)
    f.write(';')


dMMS_RS = MMS_RS.to_json(orient='index')
dMMS_RPA = MMS_i_RPA.to_json(orient='index')
dMMS_FMF = MMS_FMF.to_json(orient='index')
dMMS_RPM = MMS_RPM.to_json(orient='index')
dMMS_IMB = MMS_IMBo.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'MMS.js'), 'w') as f:
    f.write('var MMS_RS = ')
    f.write(dMMS_RS)
    f.write(';')

    f.write('var MMS_RPA = ')
    f.write(dMMS_RPA)
    f.write(';')

    f.write('var MMS_FMF = ')
    f.write(dMMS_FMF)
    f.write(';')

    f.write('var MMS_RPM = ')
    f.write(dMMS_RPM)
    f.write(';')

    f.write('var MMS_IMB = ')
    f.write(dMMS_IMB)
    f.write(';')

dMMH_RPM = MMH_RPM.to_json(orient='index')
dMMH_IMB = MMH_IMBo.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'MMH.js'), 'w') as f:
    f.write('var MMH_RPM = ')
    f.write(dMMH_RPM)
    f.write(';')

    f.write('var MMH_IMB = ')
    f.write(dMMH_IMB)
    f.write(';')


# print("Valoración lingüística por actividad")
# print("Optimista")
# print(Eval_Ling_Optimista)
jdata = Eval_Ling_Optimista.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'Eval_Ling_Optimista.js'), 'w') as f:
    f.write('var Eval_Ling_Optimista = ')
    f.write(jdata)

# print("Pesimista")
# print(Eval_Ling_Pesimista)
jdata = Eval_Ling_Pesimista.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'Eval_Ling_Pesimista.js'), 'w') as f:
    f.write('var Eval_Ling_Pesimista = ')
    f.write(jdata)

# print("Valor central")
# print(Eval_Ling_Media)
jdata = Eval_Ling_Media.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'Eval_Ling_Media.js'), 'w') as f:
    f.write('var Eval_Ling_Media = ')
    f.write(jdata)

jdata = json.dumps(VPD_Evento)
with open(os.path.join(outputDir, 'datos', 'VPD_Evento.js'), 'w') as f:
    f.write('var VPD_Evento = ')
    f.write(jdata)

# print("Valoración general del evento")
# print(EtiquetaLinguistica)
with open(os.path.join(outputDir, 'datos', 'EtiquetaLinguistica.js'), 'w') as f:
    f.write('var EtiquetaLinguistica = "')
    f.write('')
    f.write('";')

# print("Valoración lingüística de cada dimensión por actividad y evento")
# for dimension in dimensiones:
#    print(VPD[dimension, 'media'])
with open(os.path.join(outputDir, 'datos', 'dsValoraciones.js'), 'w') as f:
    f.write('var dsValoraciones = {')
    for dimension in dimensiones:
        f.write('"' + dimension + '":')
        f.write(VPD[dimension, 'media'].to_json(orient='index'))
        f.write(',')
    f.write('}')


# print(dsPrioridades)
jdata = dsPrioridades.to_json(orient='records')
with open(os.path.join(outputDir, 'datos', 'dsPrioridades.js'), 'w') as f:
    f.write('var dsPrioridades = ')
    f.write(jdata)
# print(dsEvaluacion)
jdata = dsEvaluacion.to_json(orient='records')
with open(os.path.join(outputDir, 'datos', 'dsEvaluacion.js'), 'w') as f:
    f.write('var dsEvaluacion = ')
    f.write(jdata)
# print(dsInferior)
jdata = dsInferior.to_json(orient='records')
with open(os.path.join(outputDir, 'datos', 'dsInferior.js'), 'w') as f:
    f.write('var dsInferior = ')
    f.write(jdata)
# print(dsSuperior)
jdata = dsSuperior.to_json(orient='records')
with open(os.path.join(outputDir, 'datos', 'dsSuperior.js'), 'w') as f:
    f.write('var dsSuperior = ')
    f.write(jdata)

estadisticos = {}
for actividad in actividades:
    estadisticos[actividad] = {
        'TipoActividad': rawData.loc[rawData['ACTIVIDAD'] == actividad, 'TIPO DE ACTIVIDAD'].unique()[0],
        'Num_Eval': len(rawData.loc[rawData['ACTIVIDAD'] == actividad, 'GENERO']),
        'Hombres': rawData.loc[(rawData['ACTIVIDAD'] == actividad) & (rawData['GENERO'] == 'Hombre'), 'GENERO'].count(),
        'Mujeres': rawData.loc[(rawData['ACTIVIDAD'] == actividad) & (rawData['GENERO'] == 'Mujer'), 'GENERO'].count(),
        '<15': rawData.loc[(rawData['ACTIVIDAD'] == actividad) & (rawData['EDAD'] < 15), 'GENERO'].count(),
        '15-34': rawData.loc[(rawData['ACTIVIDAD'] == actividad) & (rawData['EDAD'] > 14) & (rawData['EDAD'] < 35), 'GENERO'].count(),
        '35-69': rawData.loc[(rawData['ACTIVIDAD'] == actividad) & (rawData['EDAD'] > 35) & (rawData['EDAD'] < 70), 'GENERO'].count(),
        '>70': rawData.loc[(rawData['ACTIVIDAD'] == actividad) & (rawData['EDAD'] > 69), 'GENERO'].count(),
    }
estadisticos = pandas.DataFrame.from_dict(estadisticos, orient='index', columns=[
                                          'TipoActividad', 'Num_Eval', 'Hombres', 'Mujeres', '<15', '15-34', '35-69', '>70'])
# print(estadisticos)
with open(os.path.join(outputDir, 'datos', 'estadisticos.js'), 'w') as f:
    f.write('var estadisticos = ')
    f.write(estadisticos.to_json(orient='index'))

PAfirmativas = {}
for actividad in actividades:
    preguntas = {}
    for pregunta in rawData.filter(regex='Q').columns:
        qx = rawData.loc[rawData['ACTIVIDAD'] == actividad, pregunta]
        preguntas[pregunta] = int(qx.value_counts()['si'])

    PAfirmativas[actividad] = preguntas


PAfirmativasT = {}
for pregunta in rawData.filter(regex='Q').columns:
    PAfirmativasT[pregunta] = len(rawData[rawData[pregunta] == 'si'])

# print(PAfirmativas)
with open(os.path.join(outputDir, 'datos', 'pAfirmativas.js'), 'w') as f:
    f.write('var pAfirmativas = ')
    f.write(json.dumps(PAfirmativas))

    f.write(';var pAfirmativasT = ')
    f.write(json.dumps(PAfirmativasT))

with open(os.path.join(outputDir, 'datos', 'pvcostben.js'), 'w') as f:
    f.write('var pprefd = ')
    f.write(pesosdimensiones.to_json())

    f.write('\nvar ppard = ')
    f.write(ppdimension.to_json())

    f.write('\nvar pprefc = ')
    f.write(pesoscriterios.to_json())

    f.write('\nvar pparc = ')
    f.write(pesoparticipacion.to_json())

print("Proceso completado correctamente")
