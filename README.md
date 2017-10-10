# Selección de features para clustering de palabras
Selección supervisada y no supervisada de features para optimizar el agrupamiento de palabras según su relación sintáctica y semántica utilizando el argoritmo de k-means. 

Trabajo perteneciente a la cátedra "Minería de Datos para Texto" de Laura Alonso Alemany - FaMAF UNC. 2017
El corpus ultilizado para la selección supervisada es un dump taggeado de [Wikipedia](http://www.cs.upc.edu/~nlp/wikicorpus/).

El siguiente procedimiento pertenece a la selección supervisada de features. La selección de features no supervisada se encuentra en [Clustering de palabras, primera parte](github.com/danibosch/word_clustering)

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
5. Creación de vector de clases con los sentidos de cada palabra.
6. Vectorización de las palabras.
7. Entrenamiento supervisado para reducción de features:
   - Recursive feature elimination
   - Univariate feature selection
8. Creación de matriz final con número de ocurrencias de features seleccionados
9. Normalización de la matriz (número de ocurrencias totales de la columna sobre ocurrencias por cada fila).

### Clustering
1. Elección de número de clusters.
2. Centroides aleatorios.
3. Algoritmo de k-means usando la distancia coseno para crear los clusters.
2. Iteración a tres valores distintos de k.

## Procedimiento en detalle
Separamos el corpus en oraciones. Las oraciones se encuentran separadas por doble salto de línea.

      def parse_sents(text):
         sentences = text.split('\n\n')
         return sentences
         
Luego separamos cada oración en palabras con sus features. Cada palabra se encuentra separada por un salto de línea.

      words_in_sent = []
      for sentence in sentences:
         if len(sentence) > 10:
            words_in_sent.append(sentence.split('\n'))

A cada palabra la separamos en sus features.

      featured_words = []
      for sent in words_in_sent:
          if bool(getrandbits(1)):
              s = []
              for word in sent:
                  if not re.match("<doc", word) and not re.match("</doc", word):
                      s.append(word.split())
              featured_words.append(s)
              
Ahora recorremos la lista con todas las palabras y creamos el diccionario de features, quitando puntuaciones, números, desconocidas y creamos paralelamente el vector de clases. Cada aparición es una fila.

      lemma, pos, wclass = word[1], word[2], word[3]
      if pos[0] == 'F' or pos[0] == 'Z':
         continue
      if counts[lemma] < threshold_w:
         continue
      features = {}

Por cada aparición de la palabra agregamos su lema.

      features[lemma] = 1
      
Agregamos su POS (separadas).

      features[pos[0]] = 1
        if len(pos) > 1:
            features[pos[1]] = 1
        if len(pos) > 2:
            if not pos[2] == '0':
                features[pos[2]] = 1

Luego, de cada contexto de la palabra agregamos su lema, y su POS.

      if not word == sent[0]:
         word_before = sent[index - 1]
         if counts[word_before[1]] > threshold_c:
             izq_word = 'izq' + word_before[1]
             izq_pos = 'izq' + word_before[2][:2]
             features[izq_word] = 1
             features[izq_pos] = 1

      if not word == sent[len(sent) - 1]:
         word_after = sent[index + 1]
         if counts[word_after[1]] > threshold_c:
             der_word = 'der' + word_after[1]
             der_pos = 'der' + word_after[2][:2]
             der_sense = 'der' + word_after[3]
             features[der_word] = 1
             features[der_pos] = 1

Vectorizamos el diccionario.

      v = DictVectorizer(dtype=np.bool_, sparse=False)
      matrix_words = v.fit_transform(dict_words)
      
Aplicamos SelectFromModel. Este transformador selecciona los features según la importancia de los pesos, basado en un estimador (LinearSVC).

      lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(matrix_words, class_words)
      model = SelectFromModel(lsvc, prefit=True)
      X_new = model.transform(matrix_words)
      
Ahora creamos la matriz con cada palabra como vector, sumando los features seleccionados.

      mfinal = np.zeros((len(words_dict), X_new.shape[1]))

      for index, lemma in enumerate(words):
          i = words_dict[lemma]
          mfinal[i] += X_new[index]

Ahora normalizamos la matriz.

      normed_matrix = normalize(mfinal)
      
Aplicamos el algoritmo de clustering k-means para 3 valores de k.

      def clustering(k):
          clusterer = kmeans.KMeansClusterer(k, cosine_distance, avoid_empty_clusters=True)
          clusters = clusterer.cluster(normed_matrix, True)
          return clusters

## Resultados selección supervisada
k = 50

      ['y', 'guitarra', 'obra', 'siguiente', 'cada', 'fuerte', 'derecho', 'distribución', 'inglés', 'hacer', 'popular', 'decidir', 'capacidad', 'of', 'campo', 'por', 'ańo', 'muy', 'quedar', 'a', 'nivel', 'revista', 'principal', 'madre', 'considerar', 'su', 'importante', 'in', 'todo', 'and', 'necesario', 'la', 'poder', 'oeste', 'en', 'alguno', 'barrio', 'e', 'regresar', 'at', 'crear', 'dentro_de', 'población', 'haber', 'casa', 'así', 'entre', 'endofarticle', 'insee', 'ordenar', 'perteneciente', 'servicio', 'semana', 'paso', 'hermano', 'estar', 'ciudad', 'conocer', 'dios', 'el', 'mejor', 'guerra', 'más', 'entrar', 'ser', 'comenzar', 'así_como', 'bajo', 'sustituir', 'equipo', 'fecha', 'desde', 'cantón', 'contar', 'él', 'disco', 'estudio', 'francés', 'no', 'final', 'ejército', 'programa', 'si', 'tomar', 'también', 'para', 'que', 'luego_de', 'de', 'nuevo', 'álbum', 'ir', 'medio', 'padre', 'destacar', 'capital', 'tener', 'enlace', 'existir', 'otro', 'como', 'gran', 'rey', 'durante', 'mismo', 'ese', 'planta', 'norte', '1', 'se', 'pp.', 'realizar', 'méxico', 'externo', 'querer', 'sobre', 'hecho', 'uno', 'estadounidense', 'referencias', 'the', 'cantante', 'o', 'historia', 'grupo', 'le', 'cambiar', 'decir', 'conjunto', 'tiempo', 'película', 'hasta', 'a_través_de', 'obama', 'imagen', 'nombre']

k = 100

      ['y', 'guitarra', 'obra', 'siguiente', 'derecho', 'hacer', 'decidir', 'por', 'ańo', 'casi', 'muy', 'a', 'nivel', 'revista', 'principal', 'su', 'importante', 'todo', 'municipal', 'poder', 'en', 'técnica', 'barrio', 'e', 'at', 'crear', 'así', 'endofarticle', 'ordenar', 'perteneciente', 'servicio', 'día', 'estar', 'ciudad', 'conocer', 'el', 'poco', 'más', 'ser', 'así_como', 'bajo', 'eln', 'nacer', 'equipo', 'desde', 'cantón', 'los', 'ganar', 'él', 'disco', 'francés', 'aparecer', 'final', 'si', 'para', 'que', 'de', 'nuevo', 'medio', 'donde', 'capital', 'tener', 'enlace', 'existir', 'otro', 'como', 'rey', 'planta', 'norte', '1', 'se', 'méxico', 'externo', 'sobre', 'hecho', 'uno', 'referencias', 'o', 'historia', 'grupo', 'cambiar', 'decir', 'acabar', 'tiempo', 'película', 'hasta', 'imagen', 'con']

k = 150

      ['cierto', 'y', 'demografía', 's', 'inglés', 'cuando', 'hacer', 'decidir', 'of', 'por', 'ańo', 'a', 'principal', 'su', 'importante', 'todo', 'and', 'interpretar', 'cual', 'en', 'et', 'alguno', 'e', 'at', 'casa', 'entre', 'endofarticle', 'estado', 'hermano', 'ciudad', 'conocer', 'el', 'guerra', 'poco', 'más', 'ser', 'así_como', 'bajo', 'sustituir', 'formación', 'continuar', 'tierra', 'no', 'final', 'si', 'on', 'tomar', 'también', 'para', 'que', 'construcción', 'parte', 'de', 'nuevo', 'enlace', 'existir', 'otro', 'gran', 'rey', 'pequeńo', 'durante', 'planta', '1', 'se', 'país', 'uno', 'o', 'le', 'acabar', 'les', 'tiempo', 'película', 'hasta', 'obama', 'con', 'pieza', 'nombre']

## Resultados selección no supervisada
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
