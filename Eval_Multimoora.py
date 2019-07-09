import pandas as pd
import numpy as np

"Se carga *numpy* el archivo csv con los datos de las evaluaciones"
datos = np.genfromtxt("La_noche_2019.csv", dtype=None, names=True, delimiter = ",",encoding="utf-8")
print(datos.dtype.names)


