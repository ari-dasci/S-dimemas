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

# loading of the data, all source data is loaded, a list is created only with the scales
escala = pandas.Series(rawData['ESCALA'])
# the column of the scales is determined in order to be able to extract only the users' evaluations at a later stage.
cEscala = rawData.columns.get_loc('ESCALA')
datos = rawData.iloc[:, cEscala+1:]  # se extraen unicamente las evaluaciones
tactividades = rawData.ACTIVIDAD
# the different activities are determined by title
actividades = rawData.ACTIVIDAD.unique()
actividad_indices = {}
for actividad in actividades:
    # the indices of the corresponding scales by activity are obtained.
    indices_escalas = rawData.index[rawData['ACTIVIDAD'] == actividad].tolist()
    actividad_indices[actividad] = indices_escalas

# dataset fetching by means of regular expressions. na_action could not be applied by the pandas version.
dsPrioridades = datos.map(lambda x: int(
    re.search('(\d+)[+-]', x)[1]), na_action='ignore')
dsEvaluacion = datos.map(lambda x: re.search('[+-]', x)[0], na_action='ignore')
dsInferior = datos.map(lambda x: int(
    re.search('(\d+):', x)[1]), na_action='ignore')
dsSuperior = datos.map(lambda x: int(
    re.search(':(\d+)', x)[1]), na_action='ignore')

# print(dsInferior)
# print(dsSuperior)

# normalisation

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

# Weights and cost-benefit vector
"""
We will work with different weights, these weights will be applied for the evaluation model and for the consensus opinion of each event. 
The first weight will be calculated for each choice of criterion from 1 to n, with 1 being the first to be chosen and n the last. We take the average of each and then make an inverse distribution.
"""
# Average of each criteria
promedioscriterios = {}
for columna in dsPrioridades:
    promedioscriterios[columna] = dsPrioridades[columna].mean()
promedioscriterios = pandas.Series(promedioscriterios.copy())  # resultado

"""
To know the weight of each criterion, the inverse proportional distribution is used, which will give the highest weight to the average closest to 1 and the lowest weight to the average closest to n. Example: HIGHEST WEIGHT → 1. lowest weight → 9
"""
MCM = numpy.lcm.reduce([int(valor) for valor in promedioscriterios.tolist()])
pesoscriterios = pandas.Series(promedioscriterios.copy())
pesoscriterios = pesoscriterios.apply(lambda x: (MCM / x) * 1)
suma = pesoscriterios.sum()
pesoscriterios = pesoscriterios.apply(lambda x: (1 * x) / suma)  # resultado
sumadepesos = pesoscriterios.sum()

"""
The second weight to be calculated is the percentage of participation. That is, by the number of times a criterion was evaluated with respect to the total. The higher the number of evaluations for a criterion, the higher the weight
To calculate this weight, the criteria in any of the DataSets that were generated from the input data (rawDATA) can be collated.
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


# by dimensions
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


# lower multimoora. Using the vector of weights by importance and the vector of weights by participation, the final weight that each criterion will have is calculated.
ndp = pesoscriterios.mul(pesoparticipacion)
suma = ndp.sum()
ndp = ndp.apply(lambda x: x / suma)  # Weight normalisation result

# Arithmetic means are calculated for each of the criteria( x ), using the DataSet with the lower normalised values.
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

# Square each of the elements (x^2) of the above matrix
MMI_ma2 = MMI_ma.map(lambda x: x * x)

# The squares of each criterion are added up, and we get an array that we will use next.
MMI_suma_ma = {}
for columna in MMI_ma2:
    MMI_suma_ma[columna] = MMI_ma2[columna].sum()

# A Matrix () is generated by taking the arithmetic mean of each criterion for each event and dividing by the square root of the sum of the squares () of each criterion.
MMI_mar = pandas.DataFrame(MMI_ma.copy())
# print(MMI_mar)
for columna in MMI_mar.columns.tolist():
    for fila in actividad_indices:
        MMI_mar[columna][fila] = MMI_mar[columna][fila] / \
            numpy.sqrt(MMI_suma_ma[columna])

# Each element of each criterion in the generated Matrix is multiplied by the weight associated with that criterion using the Normalisation vector of weights. The new Matrix is (MMI_marm)
MMI_marm = pandas.DataFrame(MMI_mar.copy())
for columna in MMI_marm.columns.tolist():
    for fila in actividad_indices:
        MMI_marm[columna][fila] = MMI_mar[columna][fila] * ndp[columna]

# The Vector Ben(+)/Cost(-) is used so that the profit criteria are added and the cost criteria are subtracted.
MMI_marm_v_bc = pandas.DataFrame(MMI_marm.copy())
for columna in MMI_marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        MMI_marm_v_bc[columna][fila] = MMI_marm_v_bc[columna][fila] * \
            vectorBenCos[columna]
# print(MMI_marm_v_bc)

# To obtain the first ranking (MMI_RS), the benefit criteria are added or subtracted from the cost criteria per event. The index of the ranking is from the highest to the lowest value.
MMI_RS = {}
for fila in actividad_indices:
    MMI_RS[fila] = MMI_marm_v_bc.loc[fila].sum()
MMI_RS = pandas.Series(MMI_RS.copy())
MMI_RS = MMI_RS.to_frame(name='y_i')
MMI_RS['ranking'] = MMI_RS.rank(ascending=False)
# print(MMI_RS)

# For the following ranking, two arrays are generated, one with the maximum and one with the minimum criteria in the Matrix ( ), the maximum representing the benefit criteria and the minimum the cost criteria.
MMI_t9 = pandas.DataFrame()
MMI_t9['max'] = MMI_mar.max()
MMI_t9['min'] = MMI_mar.min()
# print(MMI_t9)

# To calculate the reference array, we will take the maximum values of each criterion that are benefit and the minimum values of cost, using the Ben(+)/Cost(-) Vector, and multiply them by the weight ( ) that each criterion has.
MMI_w_r_r_j = {}
for fila in vectorBenCos.index:
    MMI_w_r_r_j[fila] = MMI_t9['max'][fila] * \
        ndp[fila] if vectorBenCos[fila] > 0 else MMI_t9['min'][fila] * ndp[fila]
MMI_w_r_r_j = pandas.Series(MMI_w_r_r_j.copy())
# print(MMI_w_r_r_j)

# To calculate the second ranking (MMI_RPA), for each criterion value in the array ( ), the value of that criterion in the reference array ( ) is subtracted.
MMI_RPA = pandas.DataFrame(MMI_marm.copy())
for columna in MMI_RPA.columns.tolist():
    for fila in MMI_RPA.index:
        if MMI_RPA[columna][fila] != numpy.nan:
            MMI_RPA[columna][fila] = abs(
                MMI_w_r_r_j[columna] - MMI_RPA[columna][fila])
# print(MMI_RPA)

# To obtain the second ranking (MMI_RPA), the maximum value of each event is taken. The index of the ranking is the ascending order of the previous values.
MMI_i_RPA = {}
for fila in actividad_indices:
    MMI_i_RPA[fila] = MMI_RPA.loc[fila].max()
MMI_i_RPA = pandas.Series(MMI_i_RPA.copy())
MMI_i_RPA = MMI_i_RPA.to_frame(name='z_i')
MMI_i_RPA['ranking'] = MMI_i_RPA.rank()
# print(MMI_i_RPA)

# For the last ranking, each criterion is taken from the matrix ( ), and raised to a power, which is equal to the weight ( ) of each criterion. That is, criterion ^ criterion_weight.
MMI_t12 = pandas.DataFrame(MMI_mar.copy())
for columna in MMI_t12.columns.tolist():
    for fila in MMI_t12.index:
        MMI_t12[columna][fila] = numpy.power(
            MMI_t12[columna][fila], ndp[columna])
# print(MMI_t12)

# The Ben(+)/Cost(-) Vector is used so that the profit criteria are multiplied and the cost criteria are divided. As in the previous step, each element of the newly generated Matrix is taken and raised to the power corresponding to its criterion.
MMI_t13 = pandas.DataFrame(MMI_t12.copy())
for columna in MMI_t13.columns.tolist():
    for fila in MMI_t13.index:
        MMI_t13[columna][fila] = numpy.power(
            MMI_t13[columna][fila], vectorBenCos[columna])
# print(MMI_t13)

# For the third ranking (MMI_FMF), the generated matrix is used again, and all criteria for each activity are multiplied. Example Event1-Activity1 : criterion1*criterion2*criterion3. .criterionN. The index of the ranking is the descending order of the result.
MMI_FMF = {}
for fila in actividad_indices:
    MMI_FMF[fila] = MMI_t13.loc[fila].product()
MMI_FMF = pandas.Series(MMI_FMF.copy())
MMI_FMF = MMI_FMF.to_frame(name='u_i')
MMI_FMF['ranking'] = MMI_FMF.rank(ascending=False)
# print(MMI_FMF)

# To use the MMI_RPM and MMI_IMB methods to obtain the final ranking of the events, we will need the indices and their respective rankings for each activity, obtained from the three previous methods.  Matrix with the numerical rankings
MMI_rn = pandas.DataFrame()
MMI_rn['y_i'] = MMI_RS['ranking']
MMI_rn['z_i'] = MMI_i_RPA['ranking']
MMI_rn['u_i'] = MMI_FMF['ranking']
# print(MMI_rn)

# Matrix with the indexes
MMI_mi = pandas.DataFrame()
MMI_mi['y_i'] = MMI_RS['y_i']
MMI_mi['z_i'] = MMI_i_RPA['z_i']
MMI_mi['u_i'] = MMI_FMF['u_i']
# print(MMI_mi)
# The indices are taken and their squares are added up.
MMI_smi2 = {}
MMI_smi2['y_i'] = (MMI_mi['y_i'] ** 2).sum()
MMI_smi2['z_i'] = (MMI_mi['z_i'] ** 2).sum()
MMI_smi2['u_i'] = (MMI_mi['u_i'] ** 2).sum()
MMI_smi2 = pandas.Series(MMI_smi2)
# print(MMI_smi2)

# Normalised indices
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

# MMI_m are the elements to be ranked MMI_m =
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


# upper multimoor. Arithmetic means are calculated for each of the criteria( x ), using the DataSet with the lower normalised values.
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

MMS_ma2 = MMS_ma.map(lambda x: x * x)

MMS_suma_ma = {}
for columna in MMS_ma2:
    MMS_suma_ma[columna] = MMS_ma2[columna].sum()

MMS_mar = pandas.DataFrame(MMS_ma.copy())
# print(MMS_mar)
for columna in MMS_mar.columns.tolist():
    for fila in actividad_indices:
        MMS_mar[columna][fila] = MMS_mar[columna][fila] / \
            numpy.sqrt(MMS_suma_ma[columna])

MMS_marm = pandas.DataFrame(MMS_mar.copy())
for columna in MMS_marm.columns.tolist():
    for fila in actividad_indices:
        MMS_marm[columna][fila] = MMS_mar[columna][fila] * ndp[columna]

MMS_marm_v_bc = pandas.DataFrame(MMS_marm.copy())
for columna in MMS_marm_v_bc.columns.tolist():
    for fila in actividad_indices:
        MMS_marm_v_bc[columna][fila] = MMS_marm_v_bc[columna][fila] * \
            vectorBenCos[columna]
# print(MMS_marm_v_bc)

MMS_RS = {}
for fila in actividad_indices:
    MMS_RS[fila] = MMS_marm_v_bc.loc[fila].sum()
MMS_RS = pandas.Series(MMS_RS.copy())
MMS_RS = MMS_RS.to_frame(name='y_i')
MMS_RS['ranking'] = MMS_RS.rank(ascending=False)
print(MMS_RS)


MMS_t9 = pandas.DataFrame()
MMS_t9['max'] = MMS_mar.max()
MMS_t9['min'] = MMS_mar.min()
# print(MMS_t9)


MMS_w_r_r_j = {}
for fila in vectorBenCos.index:
    MMS_w_r_r_j[fila] = MMS_t9['max'][fila] * \
        ndp[fila] if vectorBenCos[fila] > 0 else MMS_t9['min'][fila] * ndp[fila]
MMS_w_r_r_j = pandas.Series(MMS_w_r_r_j.copy())
# print(MMS_w_r_r_j)

MMS_RPA = pandas.DataFrame(MMS_marm.copy())
for columna in MMS_RPA.columns.tolist():
    for fila in MMS_RPA.index:
        if MMS_RPA[columna][fila] != numpy.nan:
            MMS_RPA[columna][fila] = abs(
                MMS_w_r_r_j[columna] - MMS_RPA[columna][fila])#Corregida, faltaba valor absoluto
print(MMS_RPA)


MMS_i_RPA = {}
for fila in actividad_indices:
    MMS_i_RPA[fila] = MMS_RPA.loc[fila].max()
MMS_i_RPA = pandas.Series(MMS_i_RPA.copy())
MMS_i_RPA = MMS_i_RPA.to_frame(name='z_i')
MMS_i_RPA['ranking'] = MMS_i_RPA.rank()
# print(MMS_i_RPA)

MMS_t12 = pandas.DataFrame(MMS_mar.copy())
for columna in MMS_t12.columns.tolist():
    for fila in MMS_t12.index:
        MMS_t12[columna][fila] = numpy.power(
            MMS_t12[columna][fila], ndp[columna])
# print(MMS_t12)


MMS_t13 = pandas.DataFrame(MMS_t12.copy())
for columna in MMS_t13.columns.tolist():
    for fila in MMS_t13.index:
        MMS_t13[columna][fila] = numpy.power(
            MMS_t13[columna][fila], vectorBenCos[columna])
# print(MMS_t13)


MMS_FMF = {}
for fila in actividad_indices:
    MMS_FMF[fila] = MMS_t13.loc[fila].product()
MMS_FMF = pandas.Series(MMS_FMF.copy())
MMS_FMF = MMS_FMF.to_frame(name='u_i')
MMS_FMF['ranking'] = MMS_FMF.rank(ascending=False)
print(MMS_FMF)


MMS_rn = pandas.DataFrame()
MMS_rn['y_i'] = MMS_RS['ranking']
MMS_rn['z_i'] = MMS_i_RPA['ranking']
MMS_rn['u_i'] = MMS_FMF['ranking']
# print(MMS_rn)


MMS_mi = pandas.DataFrame()
MMS_mi['y_i'] = MMS_RS['y_i']
MMS_mi['z_i'] = MMS_i_RPA['z_i']
MMS_mi['u_i'] = MMS_FMF['u_i']
# print(MMS_mi)

MMS_smi2 = {}
MMS_smi2['y_i'] = (MMS_mi['y_i'] ** 2).sum()
MMS_smi2['z_i'] = (MMS_mi['z_i'] ** 2).sum()
MMS_smi2['u_i'] = (MMS_mi['u_i'] ** 2).sum()
MMS_smi2 = pandas.Series(MMS_smi2)
# print(MMS_smi2)


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


# By working with lower and upper limits, we can make a ranking taking into account both limits.
# For this new ranking, we will calculate the arithmetic mean of the top and bottom indices for each event.
# lower rates

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

# higher rates

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

# To use the RPM and IMB methods to obtain the final ranking of the events, we will need the indices and their respective rankings for each activity, obtained from RS, RPA and FMF.
# Matrix with numerical rankings

MMH_rn = pandas.DataFrame()
MMH_rn['y_i'] = MMH_mi_RS['ranking']
MMH_rn['z_i'] = MMH_mi_RPA['ranking']
MMH_rn['u_i'] = MMH_mi_FMF['ranking']
# print(MMH_rn)

# Matrix with the indexes
MMH_mi = pandas.DataFrame()
MMH_mi['y_i'] = MMH_mi_RS['y_i']
MMH_mi['z_i'] = MMH_mi_RPA['z_i']
MMH_mi['u_i'] = MMH_mi_FMF['u_i']
# print(MMH_mi)
# The indices are taken and their squares are added up.
MMH_smi2 = {}
MMH_smi2['y_i'] = (MMH_mi['y_i'] ** 2).sum()
MMH_smi2['z_i'] = (MMH_mi['z_i'] ** 2).sum()
MMH_smi2['u_i'] = (MMH_mi['u_i'] ** 2).sum()
MMH_smi2 = pandas.Series(MMH_smi2)
# print(MMH_smi2)

# Normalised indices
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

# MMS_m are the elements to be ranked MMS_m =
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

# Eval Ling per activity
# Using the vector of weights by importance and the vector of weights by participation, the final weight that each criterion will have is calculated.
# Importance weights = pesoscriterios
# Weights per participation = pesoparticipacion
# w^s X w^o
Eval_Ling1 = {}
Eval_LingSum = 0
for columna in pesoscriterios.index:
    Eval_Ling1[columna] = pesoscriterios[columna] * pesoparticipacion[columna]
    Eval_LingSum += Eval_Ling1[columna]  # ∑(w^s X w^o)
# Ok
# print("\n", Eval_LingSum, "\n")
# Normalisation of weights
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

# Ratings by Dim(F)
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

# Report tables

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



jdata = Eval_Ling_Optimista.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'Eval_Ling_Optimista.js'), 'w') as f:
    f.write('var Eval_Ling_Optimista = ')
    f.write(jdata)


jdata = Eval_Ling_Pesimista.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'Eval_Ling_Pesimista.js'), 'w') as f:
    f.write('var Eval_Ling_Pesimista = ')
    f.write(jdata)


jdata = Eval_Ling_Media.to_json(orient='index')
with open(os.path.join(outputDir, 'datos', 'Eval_Ling_Media.js'), 'w') as f:
    f.write('var Eval_Ling_Media = ')
    f.write(jdata)

jdata = json.dumps(VPD_Evento)
with open(os.path.join(outputDir, 'datos', 'VPD_Evento.js'), 'w') as f:
    f.write('var VPD_Evento = ')
    f.write(jdata)


with open(os.path.join(outputDir, 'datos', 'EtiquetaLinguistica.js'), 'w') as f:
    f.write('var EtiquetaLinguistica = "')
    f.write('')
    f.write('";')


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
