import pandas
import re
import numpy

#carga de los datos
rawData = pandas.read_csv('calculos.csv')#se cargan todos los datos del origen
escala = pandas.Series(rawData['ESCALA'])#se crea una lista solo con las escalas
cEscala = rawData.columns.get_loc('ESCALA')#se determina cual es la columna de las escalas para poder extraer a posterior solamente las evaluaciones de los usuarios
datos = rawData.iloc[:,cEscala+1:]#se extraen unicamente las evaluaciones
actividades = rawData.ACTIVIDAD.unique()#se determinan las diferentes actividades por titulo

#obtención de los dataset por medio de expresiones regulares
dsPrioridades = datos.applymap(lambda x: re.search('(\d+)[+-]', x)[1])
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
