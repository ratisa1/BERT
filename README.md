## BERT: Bidirectional Encoder Representations from Transformers
## BERT : Representaciones de codificador bidireccional de Transformadores  

BERT, o Bidirectional Encoder Representations from Transformers, surgió del equipo de Google AI Language, una rama de Google Research. La innovación detrás de BERT fue presentada en un artículo titulado 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', publicado en 2018 por Jacob Devlin y sus colegas.

Este método  implica preentrenar representaciones lingüísticas de propósito general en un extenso corpus de texto. Luego, el modelo resultante se utiliza para abordar diversas tareas de Procesamiento del Lenguaje Natural (PLN). A diferencia de métodos anteriores como Word2Vec, GloVe y FastText, que se entrenaban de manera unidireccional y supervisada, BERT es bidireccional. Esto significa que considera tanto el contexto a la izquierda como a la derecha de una palabra durante el procesamiento. 

Un aspecto distintivo de BERT es su enfoque no supervisado. El modelo se entrena exclusivamente utilizando un corpus de texto sin formato, como Wikipedia. Este enfoque ha demostrado ser altamente efectivo en diversas aplicaciones de NLP 

El uso de BERT tiene 2 etapas: Pre-entrenamiento y Ajuste  

1. Pre-entrenamiento: 

* Primero se elige una de las dos arquitecturas  BERT BASE o BERT LARGE  

* Se asignan 2 tareas a realizar durante el pre-entrenamiento: 
    *  Masked Language Model (MLM) : En esta tarea, se enmascaran aleatoriamente algunas palabras de una oración y se le pide al modelo que prediga las palabras enmascaradas basándose en el contexto de la oración. Se enmascaran aleatoriamente 15% de los tokens de cada secuencia WordPiece.

    *  Next Sentence Prediction (NSP): es otra tarea de pre-entrenamiento utilizada en BERT. En esta tarea, el modelo se entrena para predecir si una oración dada es la siguiente oración real que sigue a una oración de contexto dada o una oración aleatoria del corpus. Al igual que en la tarea MLM, la tarea NSP también se utiliza en la capa de entrada de la arquitectura de BERT. Específicamente, al elegir las oraciones A y B para cada ejemplo de pre-entrenamiento, el 50% del tiempo B es la siguiente oración real que sigue a A (etiquetada como IsNext), y el 50% del tiempo es una oración aleatoria del corpus (etiquetada como NotNext). El vector C se utiliza para la predicción de la siguiente oración (NSP) y no es una representación significativa de la oración sin ajuste fino, ya que se entrenó con NSP.


* Elección conjunto de datos de preentrenamiento y tokenización: Los datos de pre-entrenamiento utilizados para BERT consisten en un gran corpus de texto sin etiquetar de Internet. Específicamente, la Wikipedia en inglés (2500 millones de palabras) y el BookCorpus (800 millones de palabras) se utilizaron como corpus de preentrenamiento para BERT. El texto se limpió y tokenizó utilizando el algoritmo de tokenización de WordPieza, que permite un vocabulario de longitud variable y tokenización de subpalabras.  

* Optimización del modelo:  BERT se optimiza utilizando el algoritmo de optimización Adam con una tasa de aprendizaje de 2e-5.  


2. Fine-Tuning: 

* Conjunto de datos anotados : BERT se puede ajustar finamente en una amplia variedad de tareas de procesamiento de lenguaje natural, como la clasificación de texto, la extracción de información y la respuesta a preguntas. Para cada tarea, se utiliza un conjunto de datos anotados específico que contiene ejemplos de entrada y salida para entrenar el modelo. 

* Arquitectura del modelo preentrenado: Para seleccionar el mejor modelo BERT para una tarea dada, se pueden probar diferentes hiperparámetros del modelo, como la tasa de aprendizaje, el tamaño del lote y el número de épocas de entrenamiento, utilizando un conjunto de datos de validación. El modelo con el mejor rendimiento en el conjunto de datos de validación se selecciona como el modelo final para la tarea. 

* Fine-Tuning del modelo 

* Evaluación y ajuste : El rendimiento del modelo BERT se evalúa típicamente utilizando métricas de evaluación específicas de la tarea, como la precisión, el recall y la F1-score para la clasificación de texto, o la exactitud y el puntaje F1 para la extracción de información.  

## ARQUITECTURA  
Hay 2 tamaños del modelo: 
1. BERT BASE:
    * Cantidad de capas: 12 
    * Longitud de estados ocultos: 768
    * Cantidad de cabezales de atención: 12 
    * Parámetros totales=110 M
2. BERT LARGE:  
    * Cantidad de capas: 24
    * Longitud de estados ocultos: 1024
    * Cantidad de cabezales de atención: 16 
    * Parámetros totales=340 M  
    
La elección entre BERT BASE y BERT LARGE dependerá de la tarea específica, los recursos computacionales disponibles y el equilibrio entre rendimiento y eficiencia que se busque. En general, BERT BASE es más eficiente en términos de recursos, mientras que BERT LARGE puede proporcionar beneficios adicionales en tareas más complejas cuando los recursos lo permiten. 
### Input Embedding 
![aa.png](attachment:image\aa.png)

BERT utiliza una técnica de tokenización llamada WordPiece Tokenization. 
 
Se calcula 3 embeddings: 
* Embedding de Token: Representa la información semántica básica del token. 

* Embedding de Segmento: BERT es capaz de procesar pares de oraciones, por lo que se utiliza un embedding de segmento para distinguir entre las dos oraciones en el caso de tareas que involucran múltiples oraciones. 

* Embedding de Posición : Se encarga de representar la posición relativa de los tokens en el texto. 

Luego se suman estos 3 tipos de embeddings para cada token 