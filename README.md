**Entrenamiento**
- El entrenamiento se llevó a cabo en un notebook de GoogleColab

- De las 7 tablas del dataset se selecciona la tabla “sale_tab.txt” que contiene la variable de interés “saletime ” que contiene la fecha y la hora exacta en que un cliente realizo la compra, además de esto la tabla contiene la variable “qtysold” que corresponde al número de tiquetes que se adquieren por cada compra (máximo 8 por persona).
Para el procesamiento de los datos se inicia por filtrar la tabla “sale_tab.txt” para únicamente dejar las columnas de interés, luego se distribuyen las filas de menor a mayor para finalmente agrupar el numero de compras en días naturales.

- Se utilizo MinMaxScaler como método de regularización y se separó el dataset en 80% para entrenamiento y 20% para validación.

- Se implementó una red recurrente conformada por dos capas de unidades LSTM y una neurona de salida en una capa densa. Para que los datos ingresaran a la red, se acomodaron los datos de tal manera que ingresaran en un arreglo de 10 posiciones para realizar la predicción de valor número 11.

- Se utilizo el mse como función de perdida y una vez realizada la predicción se utiliza el MinMaxScaler para traer los datos su escala original.


**Despliegue**
- Se creo el directorio raíz “gcp_tickit” que aloja los archivos con extensión .pkl y .npy obtenidos durante el entrenamiento y que se utilizaran para realizar la predicción.

- Se utilizo Flask como framework para el despliegue en la nube (GCP).

- Se creo un entorno virtual en donde se instalaron todas las librerías necesarias.

- Se creo el archivo raíz main.py que contiene la instancia de Flask y que redirecciona el despliegue al archivo home.html donde se utiliza Bootstrap para el diseño.
