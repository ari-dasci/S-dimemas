import numpy as np
import math as ma
import pandas as pd

# Se importa el archivo csv con todas las valoraciones que se van a normalizar.
datos = pd.read_csv('La_noche_2019.csv')
# print(datos)

# Al vector de escalas se le mandaría como parámetro los números de cada una de las escalas utilizadas en la evaluación
# Nuestro modelo toma en cuenta escalas impares por que son balanceadas. En el portal se puede validar que sean impares
escalas = [3, 5, 7]
# O se puede utilizar ésta forma en donde se buscan los valores únicos dentro del DataFrame, se genera el vector y se
# se realizan las operaciones para sacar el mínimo común múltiplo. Siguiendo la restricción de las escalas impares.
escalas2 = datos['ESCALA'].unique()
#print(escalas2)

# la fórmula que se aplica es ((valoración - 1) * mcm / (escala - 1)) + 1. Donde mcm es el mínimo común múltiplo
# Por fórmula para la normalización, se toman las escalas para que sean números pares, y se calcula el Minimo Común Múltiplo
for i in range(len(escalas2)):
    escalas2[i] = escalas2[i] - 1

mcm = escalas2[0]
for i in escalas2:
    mcm = mcm * i // ma.gcd(mcm, i)
# print(mcm)

# Una vez que se tiene el mínimo común múltiplo se utiliza ése valor para aplicarlo a todos los criterios que van a ser
# normalizados. Multiplico cada uno de los datos de todos los criterios por el mínimo común múltiplo
datos.iloc[:, 7:] = (datos.iloc[:, 7:].abs() - 1) * mcm
# En el paso anterior, a cada elemento de cada criterio se le restó uno y se le multiplicó el mcm
nwd = datos.iloc[:, 6:]
# print(nwd)

# Se extraen los datos que se van a normalizar y se genera una matriz. Se toma desde la columna de ESCALA junto con cada uno
# de los criterios. Creamos un elemento vacío para guardar los resultados.
resultado = []
for index, row in nwd.iterrows():  # Se recorre la matriz
    resultado.append([])  # Cada resultado se irá agregando a la matriz de resultados
    for element in row[1:]:  # aqui inicia desde el 1 para que no divida la escala
        try:
            int(element)  # Se toma cada uno de los elementos de la fila, excepto la escala.
            resultado[index].append(
                (element / (row[0] - 1)) + 1)  # Se toma la escala (row[0]) y se le resta 1 (para que sea valor par)
            # Después, ese valor será el divisor de cada elemento de los criterios
            # Al resultado se le suma 1 y se va agregando a la fila que se esta recorriendo
            #  Quedando en la misma columna que en el frame original cada uno de los valores de los criterios
        except:
            resultado[index].append(np.nan)  # Si la operación no se puede realizar, quiere decir que es un valor
            # en blanco, por lo que se le deja la etiqueta de NaN

# Con esa matriz, creamos un nuevo dataframe.
df_result = pd.DataFrame(resultado)
# sustituimos los valores del dataframe original con los del dataframe de resultado
datos.iloc[:, 7:] = df_result.values
# En el dataframe original, le asignamos la nueva escala a la que normalizamos, que es mcm + 1
#datos.ESCALA = mcm + 1

#Exportamos el dataframe con todos los datos normalizados a una sola escala
datos.to_csv('normalizados.csv', header = True, index = False)
# print(datos)

