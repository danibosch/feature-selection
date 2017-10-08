# Selección de features para clustering de palabras
Selección supervisada y no supervisada de features para optimizar el agrupamiento de palabras según su relación sintáctica y semántica utilizando el argoritmo de k-means. 

Trabajo perteneciente a la cátedra "Minería de Datos para Texto" de Laura Alonso Alemany - FaMAF UNC. 2017
El corpus ultilizado para la selección supervisada es un dump taggeado de [Wikipedia](http://www.cs.upc.edu/~nlp/wikicorpus/).

## Procedimiento
### Procesamiento del corpus
1. Separación del corpus en oraciones y en palabras junto con sus tags.
2. Eliminación de oraciones con menos de 10 palabras.
3. Conteo de ocurrencias totales de cada palabra.
4. Creación de diccionario de palabras.
    * Agregado de diccionario de cada palabra que aparezca en el corpus, aquellas que no sean números, puntuaciones o desconocidas.
    * Agregado al diccionario de cada palabra:
        - Part-of-speech tag de la palabra.
        - Lema de la palabra.
        - Part-of-speech tag de palabra de contexto a la izquierda.
        - Part-of-speech tag de palabra de contexto a la derecha.
        - Sentido de palabra de contexto a la izquierda.
        - Sentido de palabra de contexto a la derecha.
        - Lema de palabra de contexto a la izquierda.
        - Lema de palabra de contexto a la derecha.
    * Eliminación de palabras poco frecuentes en el corpus, de los diccionarios de las palabras.
    * Eliminación de palabras poco frecuentes como contexto, de los diccionarios de las palabras.
7. Vectorización de las palabras.
8. Normalización de la matriz (número de ocurrencias totales de la columna sobre ocurrencias por cada fila).

### Clustering
1. Elección de número de clusters.
2. Centroides aleatorios.
3. Algoritmo de k-means usando la distancia coseno para crear los clusters.
2. Iteración a tres valores distintos de k.

## Procedimiento en detalle
Utilizamos la herramienta Scapy para separar en oraciones, en palabras y etiquetar cada palabra con su POS tag, morfología del tag y funcionalidad en la oración.

      nlp = spacy.load('es', vectors=False, entity=False)
      doc = nlp(dataset)
      
Las palabras pueden accederse de dos maneras:
Desde cada oración:

      doc.sents[0]
      
Desde todo el documento:

      doc[0]

Quitamos las oraciones con menos de 10 palabras:

      sents = [sent for sent in doc.sents if len(sent) > 10]

Creamos una lista con las palabras procesadas, evitando aquellas que sean puntuaciones, números o desconocidas.

      words = []
      words_lemma = []
      for sent in sents:
          for word in sent:
              if word.is_alpha:
                  words.append(word)
                  words_lemma.append(word.lemma_)

Contamos las ocurrencias totales de cada palabra en el corpus

      counts = Counter(words_lemma)
      
Usamos un diccionario de lemas, ya que el lematizador que utiliza Scapy en español sólo transforma las palabras en lowercase.
Este diccionario es una herramienta útil aunque no es precisa ya que no considera la función que cumple la palabra en la oración, lo cual genera problemas en palabras que son ambiguas.
      
      lemma_file = open("lemmatization-es.txt", "r")

Recorremos la lista de palabras descartando aquellas que ocurren pocas veces y las agregamos a un diccionario. Si la palabra fue agregada anteriormente, traemos los features para modificarlos.

      for word in words:
          w = lemmatize(word.lemma_)
          if not word.is_alpha or str.isdigit(w) or counts[w] < threshold_w:
              continue
          if not w in dicc:
              features = {}
          else:
              features = dicc[w]

Agregamos los features. 
Primero agregamos su POS tag.

      pos = "POS__" + word.pos_
      if not pos in features:
         features[pos] = 0
      features[pos] += 1
      
Luego agregamos su funcionalidad.

      dep = "DEP__" + word.dep_
      if not dep in features:
         features[dep] = 0
      features[dep] += 1

Luego, la morfología del tag, siendo ésta parseada previamente ya que se encuentran unidas en un solo string

      tags = parse_tags(word)
      for tag in tags:
         if not tag in features:
            features[tag] = 0
         features[tag] += 1
        
Agregamos los contextos, sin orden (ventana de 1 palabra).

      if not word.i == 0:
           context_izq = doc[word.i - 1]
           c_izq = lemmatize(context_izq.lemma_)
           if context_izq.is_alpha and counts[c_izq] > threshold_c:
               if str.isdigit(c_izq):
                   c_izq = "NUM__"
               if not c_izq in features:
                   features[c_izq] = 0
               features[c_izq] += 1

       if not word.i < len(doc):
           context_der = doc[word.i + 1]
           c_der = lemmatize(context_der.lemma_)
           if context_der.is_alpha and counts[c_der] > threshold_c:
               if str.isdigit(c_der):
                   c_der = "NUM__"
               if not c_der in features:
                   features[c_der] = 0
               features[c_der] += 1

Agregamos la tripla de dependencia: palabra__lemma__funcionalidad__palabra-head-del-arbol-de-dependencia-lematizada

    tripla = "TRIPLA__" + w + "__" + word.lemma_ + "__" + word.dep_ + "__" + lemmatize(word.head.lemma_)
    if not tripla in features:
        features[tripla] = 0
    features[tripla] += 1

Separamos las palabras y sus features

      features = []
      key_words = {}
      wid = 0
      for d in dicc:
          if len(d) > 0:
              key_words[d] = wid
              wid += 1
              features.append(dicc[d])
              
Vectorizamos las palabras con Sklearn

      from sklearn.feature_extraction import DictVectorizer
      v = DictVectorizer(sparse=False)
      matrix = v.fit_transform(features)
      
Normalizamos la matriz

      matrix_normed = matrix / matrix.max(axis=0)
      
Reducimos la dimensionalidad quitando aquellas columnas que tengan poca varianza, ya que no nos aporta demasiada información.

      variances = np.square(matrix_normed).mean(axis=0) - np.square(matrix_normed.mean(axis=0))
      threshold_v = 0.001
      red_matrix = np.delete(matrix_normed, np.where(variances < threshold_v), axis=1)
      
Ahora creamos los clusters:
Utilizamos el algoritmo de kmeans de nltk para crear juntar en clusters. Las semillas son aleatorias. Iteramos a 3 valores. 

      clusterer = kmeans.KMeansClusterer(k, cosine_distance, avoid_empty_clusters=True)
      clusters = clusterer.cluster(red_matrix, True)

## Resultados
k = 50

      # Stopwords
      ['cero', 'en', 'que', 'porque', 'aunque', 'john', 'durante', 'sin', 'de', 'debajo', 'disponible', 'y', 'serio', 'evo', 'laura', 'pero', 'muy', 'cometido', 'o']
      
      # Días de la semana y meses
      ['plástico', 'septiembre', 'diciembre', 'final', 'igual', 'jueves', 'domingo', 'junio', 'octubre', 'lunes', 'enero', 'abril', 'martes', 'mediodía', 'adriana', 'noviembre', 'viernes', 'sábado', 'siquiera', 'miércoles', 'setiembre', 'febrero', 'menos', 'donde', 'marzo']
      
      # Nombres propios
      ['agustín', 'marta', 'bonaerense', 'isabel', 'francisco', 'vicente', 'carlos', 'nicolás', 'graciela', 'liliana', 'guerra', 'guillermo', 'juan', 'hugo', 'alfredo', 'fe', 'martín', 'felipe', 'liberal', 'alberto', 'lorenzo', 'rosa', 'cruz', 'pedro', 'ignacio', 'jerónimo', 'ana', 'julián', 'omar', 'elettore', 'antonio']
      
      # Funcionarios públicos
      ['embajador', 'unasur', 'ong', 'camioneta', 'gerente', 'funcionario', 'decano', 'intendente', 'franco', 'marcela', 'indicador', 'jefe', 'lengua', 'vereda', 'autopista', 'arzobispo', 'viceintendente', 'vicegobernador', 'senador', 'premio', 'david', 'mandatario', 'fuero', 'presidente', 'ministro', 'piñera', 'gobernador', 'cardenal', 'vice', 'precandidato', 'comisario', 'vicepresidente', 'disposición', 'secretario', 'palacio', 'mamá', 'soledad', 'canciller', 'ministerio', 'subsecretario', 'tribunal', 'testimonio', 'bebé', 'doctor', 'fundación', 'director', 'imperio', 'colega', 'flamante', 'dictador']

k = 100

      # Stopwords
      ['también', 'hacia', 'estar', 'en', 'que', 'no', 'a', 'porque', 'por', 'aun', 'quien', 'te', 'le', 'durante', 'sin', 'de', 'y', 'luego', 'poder', 'según', 'finalmente', 'comer', 'con', 'ya', 'se', 'ser', 'muy', 'o']

      # Días de la semana y meses
      ['septiembre', 'diciembre', 'julio', 'jueves', 'domingo', 'junio', 'octubre', 'lunes', 'enero', 'martes', 'noviembre', 'viernes', 'sábado', 'miércoles', 'setiembre', 'febrero', 'donde', 'marzo']
      
      # Nombres propios
      ['blanco', 'vicente', 'eduardo', 'guerra', 'abril', 'ferreyra', 'horacio', 'alberto', 'silvia', 'pedro', 'omar', 'elettore', 'antonio']
      
      # Funcionarios públicos
      ['embajador', 'decano', 'intendente', 'jefe', 'anuncio', 'viceintendente', 'senador', 'presidente', 'ministro', 'gobernador', 'vice', 'secretario', 'coordinador', 'subsecretario', 'director', 'dictador']

k = 150

      # Stopwords
      ['también', 'estar', 'bastante', 'en', 'que', 'no', 'a', 'porque', 'por', 'tras', 'le', 'durante', 'de', 'hasta', 'y', 'poder', 'según', 'comer', 'con', 'presuntamente', 'ya', 'se', 'sino', 'cuando', 'desde', 'ser', 'o']
      
      # Días de la semana y meses
      ['septiembre', 'diciembre', 'jueves', 'domingo', 'junio', 'octubre', 'lunes', 'enero', 'martes', 'noviembre', 'viernes', 'sábado', 'miércoles', 'setiembre', 'febrero', 'marzo']
      
      # Nombres propios
      ['vicente', 'eduardo', 'costa', 'abril', 'ceballos', 'rafael', 'alberto', 'marcelo', 'silvia', 'pedro', 'omar', 'elettore', 'antonio']

      # Funcionarios públicos
      ['embajador', 'decano', 'intendente', 'fiscal', 'impuesto', 'jefe', 'anuncio', 'arzobispo', 'viceintendente', 'vicegobernador', 'senador', 'presidente', 'ministro', 'teatro', 'gobernador', 'cardenal', 'vice', 'comisario', 'secretario', 'subsecretario', 'bebé', 'director', 'imperio', 'flamante', 'dictador']

## Instalación
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
    $ conda create --name keras python=3.5
    $ source activate keras
    (keras) $ conda install --yes --file requirements.txt
    (keras) $ python -m spacy download es
    (keras) $ export tfBinaryURL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl
    (keras) $ pip install $tfBinaryURL
    (keras) $ conda install -c conda-forge keras
    (keras) $ jupyter notebook

    KERAS_BACKEND=tensorflow jupyter notebook
