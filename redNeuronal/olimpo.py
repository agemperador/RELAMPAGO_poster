
"""""""""""""""LIBRERIAS DE MACHINE LEARNING"""""""""""""""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#### LIBRERIA PARA MANEJO DE IMAGENES ####
from PIL import Image

#### LIBRERIA PARA HACER LA IMAGEN DE FRECUENCIA DE PALABRAS #### 
from wordcloud import WordCloud

from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------ZEUS-------------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""CARGA DEL DATASET DE TRAIN Y TEST"""""""""""""""
def zeus(archivo = 'base_corr_orto - Copy of base_corr.csv', Data_path = './tweets.csv',
        name_MSJ = 'Twit',name_TAG = 'Tag', name_NUM = 'Num',name_USER = 'user_id',name_csv = 'TW_filtrados'):
    """
    Función principal que conecta todas las funciones de procesamiento de los datos y generación del Dataframe final.
    Carga el archivo csv, le agrega una columna de Usuario para luego filtrarlos con la función hermes.
    Luego pasa el dataset por la función dionisio donde se normaliza puliendo cosas innesesarias.
    
    This function provides de conection of all the first part of the data analysis.
    It begins loading both, the csv of tagged twits and the csv of all the twits.
    Then it use those to merge in one dataframe with a new column named USER and the 'hermes' function is called.
    The hermes function will make a filter by the 'spam' users removing those who have more than a number_spam of twits.
    After that, it creates a new column on de dataframe and saves de normalization of the Message column on the MSJ_NORM column with the 'dioni' function.
    At the end the dataframe will be saved on a new csv with the name of 'name_csv'.
    ------------------------------------
    Datos de entrada:
    ------------------------------------
    archivo = path + nombre del csv
    Data_path
    name_MSJ = Nombre de la columna de Twit
    name_TAG = Nombre de la columna de Tags
    name_NUM = Nombre de la columna de número de twit
    name_csv = Name of the new csv that will be created
    ------------------------------------
    Datos de salida:
    ------------------------------------
    TW = Is the Dataframe zeuzed 
    
    """
    DATA = pd.read_csv(Data_path,sep=';')
    print('Archivo crudo cargado de %s'%Data_path)
    
    ########################################################
    """""""""""""""SE CARGA EL CSV CORREGIDO"""""""""""""""
    ########################################################
    TW = pd.read_csv(archivo)
    print('Datos corregidos cargados de "%s"' %archivo)

    MSJ = TW[name_MSJ]
    TAG = TW[name_TAG]
    NUM = TW[name_NUM]
    WHO = TW['Nombre']
    USER = DATA[name_USER][np.asarray(NUM)]

    FINAL = pd.DataFrame(list(zip(MSJ, TAG, NUM, USER, WHO)), columns = ['MSJ', 'TAG', 'NUM', 'USER', 'WHO'])
    
    TW = hermes(FINAL,name_USER='USER')

    
    ########################################################
    """""""""""""""""NORMALIZAMOS LOS TWITS"""""""""""""""""
    ########################################################
    
    print(len(TW['MSJ']), len(FINAL['MSJ']))
    
    TW['MSJ_NORM'] = dioni(TW['MSJ'])
    
    
    TW.to_csv(r'%s.csv'%name_csv, index = None, sep = ';')

    ##### ACÁ VA LA CORRECCIÓN ORTOGRAFICA ###
    
    return TW

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------DIONI------------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# EX normalizacion
def dioni(mensaje, emoticones = False):
    """
    Esta función toma un dataset de twits y los normaliza.
    Devuelve: mensajes normalizados, el diccionario de palabras y la frecuencia de palabras (En ese orden).
    Aplica minúsculas.
    Quita los emoticones.
    Arregla los problemas con acentos.
    Elimina los simbolos especiales.
    Quita las palabras no deseadas
    
    """
    
    mensaje = list(mensaje)
    lenght = len(mensaje)
    
    #############################################################################################################
    """""""""""""""""""""""""""""""""""PALABRAS QUE HAY QUE CAMBIAR POR ASCII"""""""""""""""""""""""""""""""""""
    #############################################################################################################
    lista_ascii = [['\\xe1', '\\xe9', '\\xed', '\\xf3', '\\xfa', '\\xf1', '\\xd1', '\\xc1','\\xc9', '\\xcd', '\\xd3', 
                 '\\xda', '\\xa1', '\\xbf', '\\xdc', '\\xfc', '\\xb4', '\\xba', '\\xaa', '\\xb7','\\xc7','\\xe7',
                 '\\xa8', '\\xb0C', '\\n', '\\xb0c', '\\xbb', 'xb0', '\\ufe0f'],
                   ['a', 'e', 'i', 'o', 'u', 'ñ', 'Ñ', 'A', 'E', 'I', 'O', 'U', '', '', 'Ü', 'ü', '', 
                           ' ', '', '','Ç','ç','', ' ', ' ', ' ', ' ', ' ', ' ']]
    
    #############################################################################################################
    """""""""""""""""""""""""""""""""""""""""""""SIMBOLOS QUE VUELAN"""""""""""""""""""""""""""""""""""""""""""""
    #############################################################################################################
    lista_simb = ['!','\'', '\"', "|", '$', "%", "&", "(", ")", "=", "?", "+",'/', ";", "_", "-", "1","2",
                  "3","4","5","6","7","8","9","0", '*',',', '.', ':', '#']
    
    #############################################################################################################
    """""""""""""""""""""""""""""""""""""""""""""""PALABRAS A IGNORAR"""""""""""""""""""""""""""""""""""""""""""""""
    #############################################################################################################
    ign_palabras = ['mbar','cdu','s','la','lo','st','que', 'me','t','p','el','weathercloud','en','h',       
                    'temp','hpa','km','mm',"su","vos",'que',"re","d","xq", "le","te","tu" ,"soy"                                                                                                                                                                            
                    ,"sos","mi","da","o","x","les","me","d", "q", "lo", "los", "mi", "son", "a", "el", 
                    "un","la", "una","en","por","para", 'las',"ante", "al", 'me',"rt", "del", "y", "de",
                    "que", "sus", "ha", "es", "con", "nos", "eh", "xd"]

    for i in range(lenght):
        
        ## Convierto mayúsculas en minúsculas
        mensaje[i] = mensaje[i].lower()

        ## Saco las menciones y otras cosas
        txt = mensaje[i].split()
        
        # Elimina las palabras con @, elimina todo lo que contenga "jaj" y todos los links
        # Menos los @RELAMPAGO2018 O @RELAMPAGO_edu
        for j in range(len(txt)):
            if ('@' in txt[j]) and ('RELAMPAGO2018' not in txt[j]) and ('RELAMPAGO_edu' not in txt[j]) or ('jaj' in txt[j]) or ('https' in txt[j]): txt[j]=''
        
        # Une las palabras para cada twit
        mensaje[i] = " ".join(txt)
        
        ## Reemplazo símbolos
        for h in range(len(lista_simb)):
            mensaje[i] = mensaje[i].replace(lista_simb[h], ' ')

        
        
        ## Convierto el msj a ASCII, reemplazo los errores de decodificación y agrego un espacio antes de cada decodificación
        mensaje[i] = mensaje[i].encode('unicode-escape').decode('ASCII')+" "     
        
        mensaje[i] = mensaje[i].lower()
        # Esto quita los emoticones
       
        ## Reemplazamos los ascii 
        for k in range(len(lista_ascii[0])):
            mensaje[i] = mensaje[i].replace(lista_ascii[0][k], lista_ascii[1][k])

        ## Para separar emojis
        mensaje[i] = mensaje[i].replace('\\', ' \\')
        
                
        if mensaje[i][0] != ' ': 
            mensaje[i] = ' ' + mensaje[i]
        
        ## ignoramos
        for j in ign_palabras:
            mensaje[i] = mensaje[i].replace(" "+j+" ", ' ')
        
    return mensaje

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------AFRODITA---------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""IMAGEN CHETA"""""""""""""""
#EX makeIMAGE
def afrodita(text, max_words = 200,width=1920, height=1080):
    """
    Esta función es la que genera la imagen de frecuencias:
    Se ingresan toda la base de twits en forma de frecuencias, generadas por la función zeus.
    text = frecuencias
    maxwords = cantidad máxima de palabras a mostrar.
    
    """
    # Para que tenga forma de alicia
    #alice_mask = np.array(Image.open("cord.jpeg"))

    wc = WordCloud(background_color="white", max_words=max_words,contour_color ='white', width=width, height=height) #, mask=alice_mask)
    # Esto genera la nube de palabras
    wc.generate_from_frequencies(text)

    # show
    fig = plt.figure(figsize=(30,10))
    plt.imshow(wc, interpolation="bilinear",cmap=plt.get_cmap("coolwarm"))
    #plt.axis("off")
    plt.tick_params(bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft = False)
    plt.savefig('Multiword')
    plt.box(on=None)
    plt.show()

    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------ATENAS-----------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
#################################################################
"""""""""""""""""""""ENTRENAMIENTO DE LA RED"""""""""""""""""""""
#################################################################
def atenas(train,word_to_ix, label_to_ix,capas = [], 
           lr = 0.1, epocas = 20, err = 0.01, opt = 'sgd', nombre_modelo = 'BOWCLASS' ):
    """
    Función de entrenamiento de la red neuronal.
    Primero instancia un modelo BowClassifier con los sizes
    Luego genera el entrenamiento con SGD y segund la clase instanciada
    Por ultimo, genera los gráficos del error y devuelve el modelo
    
    --------------------------------------------
    Datos de entrada:
    --------------------------------------------
    train = Datos de entrenamiento 
    word_to_ix = Diccionario de palabras
    label_to_ix = Diccionario de Tags
    capas = Lista de los tamaños de las capas ocultas 
    lr = Learning rate del entrenamiento
    epocas = Cantidad de epocas a entrenar
    err = Error límite para cortar el entrenamiento
    --------------------------------------------
    Datos de salida:
    --------------------------------------------
    model = La instancia del modelo entrenado
    
    
    
    """
    
    
    print('Entrenando con un lr = %f, hasta %i epocas error mínimo de: %f' %(lr,epocas,err))
    ## Instancia un el modelo con tamaño sizes

    ## Capa de entrada va de VOCAB_SIZE a N 
    ## Capa oculta de N a NUM_LABELS
    #Hace una lista con los tamaños de las capas
    VOCAB_SIZE = len(word_to_ix)
    NUM_LABELS = len(label_to_ix)

    if capas == [-1]: capas = []
    capas.append(NUM_LABELS)
    capas.insert(0,VOCAB_SIZE)

    sizes = capas
    if nombre_modelo == 'BOWCLASS':
        model = BoWClassifier(sizes)
    elif nombre_modelo == 'EMBEDDING':
        model = RNNClassifier(VOCAB_SIZE,200)
    loss_function = nn.CrossEntropyLoss()
    
    if opt == 'adam': optimizer = optim.Adam(model.parameters(), lr)
    else: optimizer = optim.SGD(model.parameters(), lr)
    # Se suele usar la cantidad de epocas que lleve a la estabilidad sin overfitear

    model.train()
    error = []

    epoch, e = 0,1
    while epoch <= epocas and e >= err:
        #if epoch % 10 == 0: print('Epoch: %i' %epoch)
        for instance, label in train:

            if instance == ['nan'] or str(label) == 'nan': continue
            
            #if instance == 'nan': continue
            # Hay que resetear los gradientes antes de hacer cuentas
            model.zero_grad()

            # Hacemos el vector de bag of words
            
            bow_vec = BoWClassifier.make_bow_vector(instance, word_to_ix)

            # Guardamos el label del diccionario de Tags
            
            target = BoWClassifier.make_target(label, label_to_ix)

            # Hacemos el forward pass.
            log_probs = model(bow_vec)

            # Calculamos el error
            loss = loss_function(log_probs, target)

            # Le pedimos a torch que busque los gradientes con back propagation

            loss.backward()

            # Usamos SGD para hacer back propagation y modificar los pesos
            optimizer.step()
        e = loss
        error.append(loss) 
        print('Epoch: %i %f' %(epoch,error[-1]))
        epoch +=1


    #######################
    """GRÁFICO DEL ERROR"""
    #######################
    plt.figure()
    plt.plot(error)
    plt.show()
    
    
    return model



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------CRONOS-----------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def cronos(test,word_to_ix, model):
    """
    Función que genera la verificación de todo.
    Toma el modelo entrenado y utiliza el test para ver los aciertos
    Luego muestra los aciertos si y no y las sorpresas y falsos positivos.
    Por ultimo genera dos gráficos de barras en los que se muestra la información obtenida
    -----------------------------------
    Datos de entrada:
    -----------------------------------
    test = Datos de test
    word_to_ix = Diccionario de palabras
    model = Modelo entrenado
    -----------------------------------
    Datos de salida:
    -----------------------------------
    Meteo = Acierto de que el twit ES meteorológicos
    NoMeteo = Acierto de que el twit NO ES meteorológico
    Sorpresas = Falla en la que el modelo dijo que no era pero en verdad SI fue meteorológico
    Falsos = Falla en la que el modijo que era, pero NO fué meteorológico
    total = Total de twits analizados
    """
    
    # Inicializo en 0
    Meteo ,NoMeteo, Sorpresas, Falsos, total = 0,0,0,0,0
    model.eval()
    for instance, label in test:
        if instance == ['nan']: continue
        bow_vec = autograd.Variable(BoWClassifier.make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)
        log_probs_exp = log_probs.exp()
        if bool (log_probs_exp[0][0]>0.5):

            if label == 1: 
                Meteo+=1
                total +=1
               # print ('\033[1;93m' , log_probs_exp , label)
            else: 
                Falsos+=1
                total +=1
               # print ('\033[1;31m' , log_probs_exp , label)

        elif bool (log_probs_exp[0][1]>0.5):

            if label == 0:
                NoMeteo+=1
                total +=1
               # print ('\033[1;32m' , log_probs_exp , label)

            else:
                Sorpresas+=1
                total +=1
               # print ('\033[1;66m' , log_probs_exp , label)


    print('\033[0m Aciertos a tweets meteorológicos:', round((Meteo/total)*100, 3),
          '% \n Aciertos a tweets no meteorologicos:', round((NoMeteo/total)*100, 3),
          '% \n Sorpresas: ', round((Sorpresas/total)*100, 3),
          '% \n Falsos positivos: ', round((Falsos/total)*100, 3), '%') 

    fig, ax = plt.subplots(figsize=(12,9))
    #plt.rcParams['font.sans-serif'] = 'Arial'
    #plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = 'black'
    #plt.rcParams['axes.labelcolor']= '#909090'
    plt.rcParams['xtick.color'] = '#909090'
    plt.rcParams['ytick.color'] = '#909090'
    plt.rcParams['font.size']=23
    labels = ['Success', 
             'Surprises',
             'False positives']
    
    percentages = [Meteo + NoMeteo,Sorpresas,Falsos]
    
    color_palette_list = ['#55C667FF', '#33638DFF','#440154FF' , '#0EBFE9',   
                          '#C1F0F6', '#0099CC']
    explode=(0.1,0.,0.)
    ax.pie(percentages, explode=explode,   
           colors=color_palette_list[0:3], autopct='%1.0f%%', 
           shadow=False, startangle=0,   
           pctdistance=1.2,labeldistance=1.4)
    ax.axis('equal')
    #ax.set_title("Elephant in the Valley Survey Respondent Make-up")
    ax.legend(labels[0:3],frameon=False, bbox_to_anchor=(1.5,0.8))
    
    plt.show()
    return Meteo,NoMeteo,Sorpresas,Falsos,total

    #####################################################
    """""""""""""""GRÁFICO DE VERIFICACIÓN"""""""""""""""
    #####################################################
"""    
    fig = plt.figure(figsize=(14,11))
    opacity = 0.4
    bar_width = 0.35

    ax = plt.subplot(1,2,1)

    bar = plt.bar(range(4),[Meteo, NoMeteo,Falsos,Sorpresas],
                  tick_label=['Meteorological \n success','Not Meteorological \n success',
                              'False \n Positive','Surprises'], color = 'orange')
    
    plt.title('Results', fontsize = 24)
    plt.ylabel('Successes.  Total='+str(total))
    for rect, porcentaje in zip(bar,[Meteo,NoMeteo,Falsos,Sorpresas]):
        height = rect.get_height()- 25
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % round((porcentaje/total)*100, 3) + ' %'
                 ,fontsize = 14, color = 'black',ha='center', va='bottom')
        
    #################################################
    COMPARACIÓN DE TAGS
    #################################################
    ax2 = plt.subplot(1,2,2)

    #color=plt.cm.viridis_r(range(0,300,round(300/4)))

    bar1 = plt.bar(np.arange(2), [Meteo + Sorpresas,NoMeteo + Falsos], bar_width, 
                   tick_label = ['Meteorological', 'Not Meteorological'],
                   color = '#55C667FF', label = 'Total of Twits'
                  )

    for rect, porcentaje in zip(bar1,[Meteo + Sorpresas,NoMeteo + Falsos]):
        height = rect.get_height()-25
        plt.text(rect.get_x() + rect.get_width()/2, height, '%d' % porcentaje
                 ,fontsize = 14, color = 'white',ha='center', va='bottom')

    color=plt.cm.viridis(range(0,300,round(300/4)))
    
    bar2 = plt.bar(np.arange(2) +bar_width*1.1, [Sorpresas,Falsos], bar_width,
                  tick_label = ['Meteorological', 'Not Meteorological'],
                   color = '#440154FF' , label= 'Total Fails'
                  )
    for rect, porcentaje in zip(bar2,[Sorpresas/(Meteo + Sorpresas),Falsos/(NoMeteo + Falsos)]):
        height = rect.get_height()-25
        plt.text(rect.get_x()+rect.get_width()/2 , height, '%d %%' % round(porcentaje*100,3)
                 ,fontsize = 14, color = 'white',ha='center', va='bottom')

    plt.legend()
    plt.title('Comparison', fontsize = 24)
    plt.savefig('Comparativa.png')
    plt.show()
"""    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------ALEXANDRIA-------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def alexandria(mensaje):
    ## Hay que hacerlo lista porque es un pandas dataframe
    mensaje = list(mensaje)
    
    word_to_ix = {}
    word_cant = {}
    
    for i in range(len(mensaje)):
        """""""""DICCIONARIO"""""""""
           #Diccionario y frecuencias
        #print(mensaje[i])
        if str(mensaje[i]) == 'nan':
            continue
        for word in mensaje[i].split():
            
            # El diccionario tiene que tener todas las palabras posibles
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
                word_cant[word]  = 1
            else: word_cant[word]+=1
            
    return word_to_ix,word_cant
    

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------HERMES-----------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def hermes(DATAFRAME, name_USER = 'USER', number_spam = 5 ):
    cant_users = list(set(DATAFRAME[name_USER]))
    
    spam = []
    for i in cant_users:
        h = list(DATAFRAME[name_USER]).count(i)
        if h >= number_spam: spam.append(i)
    
    aux = []       
    for i in spam:
        aux = np.append(aux, np.where(DATAFRAME[name_USER] == i)[0])
    
    DATAFRAME = DATAFRAME.drop(aux, axis = 0)
        
    return DATAFRAME

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#-----------------------------SOCRATES---------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Función que carga el train y test
def socrates(MSJ ,TAG, size = 0.8, emoticones= False):   
    """
    Función que toma los twits crudos y devuelve los datos en train y test.
    Toma los twits de un dataframe
    Los pasa por la función dioni (de dionisio) que los normaliza.
    Luego, divide los mensajes en train y test con un tamaño 'size'
    Devuelve los datos train, test y los mensajes, diccionario y frecuencias.
    
    Los datos de entrada son:
    
    TW = Dataframe de twits
    MSJ = Nombre de la columna de twits
    TAG = Nombre de la columna de Tags
    NUM = Nombre de la columna de número de twit
    size = tamaño del train va de 0 a 1
    emoticones = bool que dice si se toman en cuenta los emoticones o no.
    
    Devuelve:
    train, test = una permutacion aleatoria segun size que tiene tamaño 2xN con N tamaño de train o test
    msj,dic y frec es lo que devuleve la función dioni.
    
    """
    """"""""""""""
        
    print('Tamaño de los mensajes: %i' %len(MSJ))

    
    print('Generando TRAIN Y TEST:')
    
    #### Genero el random para el train y test ####
    randordtag = [x for x in sorted(TAG)]
    randordmsj = [x for _,x in sorted(zip(TAG,MSJ))]

    randtw_si = np.array(randordmsj[-len(TAG[TAG==1]):])[torch.randperm(len(TAG[TAG==1])).numpy()]
    
    randtw_no = np.array(randordmsj[:len(TAG[TAG==0])])[torch.randperm(len(TAG[TAG==0])).numpy()]
    

    leng = int(size*min(len(TAG[TAG==1]),len(TAG[TAG==0])))
    
    print('Tamaño de cada tag: %i' %leng)

    #Permutación al azar que se usa para los dos tags
    perm = torch.randperm(leng).numpy()

    #junto las listas de twits y genero los tags
    rand_testtw = np.append(randtw_si[leng:],randtw_no[leng:])
    rand_testtag = np.append(np.zeros(len(randtw_no)-leng),np.ones(len(randtw_si)-leng))


    randtw_si = randtw_si[perm]
    randtw_no = randtw_no[perm]
    
    randtw = np.append(randtw_no, randtw_si)
    
    # Si no le agrego el dtype no le gusta
    randtag = np.append(np.zeros(len(randtw_no),dtype = int),np.ones_like(randtw_si))

    # Permutación para la lista final
    perm = torch.randperm(len(randtw)).numpy()
    
    #Randomizo la lista final    
    randtw = randtw[perm]
    randtag = randtag[perm]

    #########################################################
    """ARREGLAR ESTO"""
    #########################################################
    train = [(i.split(),j) for i,j in zip(randtw,randtag)]
    test = [(i.split(),j) for i,j in zip(rand_testtw,rand_testtag)]

    return train, test

    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#---------------------------RED BOW CLASS------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class BoWClassifier(nn.Module): # inheriting from nn.Module!
    """
    Clase que abarca a la red y las funciones necesarias
    """
    
    def __init__(self, sizes):
        """
        Función que genera la estructura de la red con las capas pedidas en sizes
        
        """
        
        super(BoWClassifier, self).__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
        
        
    def forward(self, bow_vec):
        """
        Función que pasa los bow_vectors por la red
        Devuelve la salida de la red
        """
        
        h = bow_vec
        
        # Itero todas las capas menos la ultima
        for hidden in self.layers[:-1]:
            h = torch.tanh (hidden (h))
        #La ultima es output
        output = self.layers[-1]
        y = output(h)
        #La uso con softmax
        
        #soft = F.log_softmax(y , dim=1)
        
        return y
    
    def make_bow_vector(sentence, word_to_ix):
        """
        Función que devuelve en base a una frase, el bag of words correspondiente segun el diccionario.
        """
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            
            vec[word_to_ix[word]] += 1
        return vec.view(1, -1)

    def make_target(label, label_to_ix):
        
        return torch.LongTensor([label_to_ix[label]])
    
    
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##################################################################
#----------------------------------------------------------------#
#---------------------------RED RNN CLASS------------------------#
#----------------------------------------------------------------#
##################################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class RNNClassifier(nn.Module): # inheriting from nn.Module!
    """
    Clase que abarca a la red y las funciones necesarias
    """
    
    def __init__(self, vocab, embedding,hsize=10, sizes=[0]):
        """
        Función que genera la estructura de la red con las capas pedidas en sizes
        
        """
        super(RNNClassifier, self).__init__()

        self.isize = vocab
        self.esize = embedding
        
        self.embedding =  torch.nn.Embedding(self.isize, self.esize)     
 
        self.linear1 = torch.nn.Linear(self.esize, hsize)

        self.linear2 = torch.nn.Linear(hsize, 2)

        self.emb = torch.nn.Embedding(self.isize,self.esize)
        self.layers = torch.nn.ModuleList()

        for i in range(len(sizes)-1):
            self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
        
        
    def forward(self, bow_vec):
        """
        Función que pasa los bow_vectors por la red
        Devuelve la salida de la red
        """
        
        h = bow_vec
        h = self.embedding(h)
        
        out = torch.tanh(self.linear1(h))
        
        # Itero todas las capas menos la ultima
        #for hidden in self.layers[:-1]:
        #    h = torch.tanh (hidden (h))
        #La ultima es output
        #output = self.layers[-1]
        output = torch.tanh(self.linear2(h))
        #y = output(h)
        #La uso con softmax
        
        #soft = F.log_softmax(y , dim=1)
        
        return output
    
    def make_bow_vector(sentence, word_to_ix):
        """
        Función que devuelve en base a una frase, el bag of words correspondiente segun el diccionario.
        """
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            
            vec[word_to_ix[word]] += 1
        return vec.view(1, -1)

    def make_target(label, label_to_ix):
        
        return torch.LongTensor([label_to_ix[label]])  
    
class EmbeddingTrain(nn.Module):
    
    def __init__(_, vocab, esize =200):
        
        super().__init__()
        
        _.isize = vocab
        _.embsize = esize
        
        _.emb = nn.Embedding(_.isize,_.embsize)
    
        
    def forward(_, x):
        """
        Función que pasa los bow_vectors por la red
        Devuelve la salida de la red
        """
        
        output = _.emb(x)
        
                
        return output
    
    def one_hot(word,word_to_ix):
        
        vec = torch.zeros(len(word_to_ix))
        vec[word_to_ix[word]] +=1
        
        return vec.view(1,-1)
    
    def make_bow_vector(sentence, word_to_ix):
        """
        Función que devuelve en base a una frase, el bag of words correspondiente segun el diccionario.
        """
        vec = torch.zeros(len(word_to_ix))
        for word in sentence:
            
            vec[word_to_ix[word]] += 1
        return vec.view(1, -1)

    def make_target(label, label_to_ix):
        
        return torch.LongTensor([label_to_ix[label]])  
        
        
        
        