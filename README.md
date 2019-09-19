# Meteorología y redes sociales: aplicación de Redes Neuronales para la detección de fenómenos severos.

## Presentación

El proyecto RELAMPAGO-CACTI (https://sites.google.com/illinois.edu/relampago/home) se desarrolló durante la segunda mitad del año 2018 y principios del años 2019 en las provincias de Córdoba y Mendoza, Argentina, con el objetivo de estudiar las tormentas convectivas generadas en la zona. La función de uno de los tantos equipos que formaron parte del proyecto era registrar los eventos severos reportados por el público, principales afectados por las inclemencias climáticas. La red social Twitter fue la plataforma elegida para este fin: los perfiles abiertos al público y la utilización de palabras clave (tipos de eventos meteorológicos y ubicación) permitieron generar una base de datos de reportes ciudadanos con la cual fue posible la verificación de ocurrencia de fenómenos severos (http://catalog.eol.ucar.edu/relampago/loop/259786?date=2018/11/15&count=24). El proyecto finalizó formalmente en Abril de 2019, hasta ese mes se obtuvieron más de 600 reportes verificados.

El trabajo de la recolección de tweets fue realizado a mano por un equipo de 9 estudiantes. Sin embargo, el trabajo puede seguir y mejorarse automatizando la obtención de la base de datos mediante la utilización de una red neuronal capaz de diferenciar tweets meteorológicos de no meteorológicos.

## Objetivo
Este trabajo se divide en dos grandes ejes:

### 1) La base de datos:
Es necesario generar una buena base de datos en base a publicaciones de la red social Twitter, compuesta tanto por tweets meteorológicos como no meteorológicos. Esta base de datos se utilizará como datos de train y test en la red neuronal. La clasificación de los tweets se realizó individualmente y de forma manual.

### 2) Un buen entendimiento automático:
Crear una red neuronal que a partir de la base de datos generada anteriormente logre diferenciar tweets meteorológicos de no meteorológicos al encontrarse con una nueva base de datos no clasificada. Esta herramienta debería ser capaz de automatizar la detección de tweets meteorológicos sin necesidad de interacción humana.

