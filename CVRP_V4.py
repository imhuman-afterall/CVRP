import numpy as np
import matplotlib.pyplot as plt
import random
import time

# -------------------------------------------------------------------------------
# File reader
with open("P-n16-k8.vrp.txt", "r") as file:
    instancia_texto = file.read()

def procesar_instancia(instancia_texto):
    coordenadas = []
    demandas = []
    procesando_nodos = False
    procesando_demandas = False

    for linea in instancia_texto.split("\n"):
        if linea.strip() == "NODE_COORD_SECTION":
            procesando_nodos = True
            procesando_demandas = False
            continue
        elif linea.strip() == "DEMAND_SECTION":
            procesando_nodos = False
            procesando_demandas = True
            continue
        elif linea.strip() == "DEPOT_SECTION":
            break

        if procesando_nodos:
            partes = linea.split()
            coordenadas.append((float(partes[1]), float(partes[2])))
        elif procesando_demandas:
            partes = linea.split()
            demandas.append(int(partes[1]))

    return coordenadas, demandas

# -------------------------------------------------------------------------------
# Distance matrix
def calcular_matriz_distancias(coordenadas):
    num_nodos = len(coordenadas)
    matriz_distancias = np.zeros((num_nodos, num_nodos))

    for i in range(num_nodos):
        for j in range(num_nodos):
            x1, y1 = coordenadas[i]
            x2, y2 = coordenadas[j]
            matriz_distancias[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return matriz_distancias

# -------------------------------------------------------------------------------
# ******************* PHASE ONE - NEAREST NEIGHBOR ALGORITHM ********************
# -------------------------------------------------------------------------------
def vecino_mas_cercano(matriz_distancias, demandas, capacidad, k=3):
    num_nodos = matriz_distancias.shape[0]
    rutas = []
    ruta_costo = []
    tot_costo = 0
    used_q = []

    visitado = np.zeros(num_nodos, dtype=bool)
    visitado[0] = True

    while np.sum(visitado) < num_nodos:
        nodo_actual = 0
        capacidad_actual = 0
        distancia_ruta = 0
        ruta = [nodo_actual]

        while True:
            vecinos_candidatos = []

            for vecino in range(1, num_nodos):
                if not visitado[vecino] and demandas[vecino] + capacidad_actual <= capacidad:
                    distancia = matriz_distancias[nodo_actual, vecino]
                    vecinos_candidatos.append((vecino, distancia))

            if not vecinos_candidatos:
                break

            vecinos_candidatos.sort(key=lambda x: x[1])
            k_mejores = vecinos_candidatos[:min(k, len(vecinos_candidatos))]
            mas_cercano, min_distancia = random.choice(k_mejores)

            ruta.append(mas_cercano)
            visitado[mas_cercano] = True
            capacidad_actual += demandas[mas_cercano]
            distancia_ruta += min_distancia
            nodo_actual = mas_cercano

        distancia_regreso = matriz_distancias[ruta[-1], 0]
        distancia_ruta += distancia_regreso
        ruta.append(0)

        rutas.append(ruta)
        ruta_costo.append(distancia_ruta)
        used_q.append(capacidad_actual)
        tot_costo += distancia_ruta

        if np.sum(visitado) < num_nodos:
            for i in range(1, num_nodos):
                if not visitado[i]:
                    nodo_actual = i
                    break

    return rutas, ruta_costo, tot_costo, used_q

# -------------------------------------------------------------------------------
# Extra processing
coordenadas, demandas = procesar_instancia(instancia_texto)
matriz_distancias = calcular_matriz_distancias(coordenadas)
rutas, ruta_costo, tot_costo, used_q = vecino_mas_cercano(matriz_distancias, demandas, 35, k=3)

# Saving results as a new file
with open("Results.txt", "w") as f:
    f.write('==== PHASE ONE :: RESULTS - k-NEAREST NEIGHBOR ALGORITHM ==== \n')
    for i, ruta in enumerate(rutas, 1):
        f.write(f'R{i}: {ruta} \n\tCOST: {ruta_costo[i-1]} -- CAPACITY: {used_q[i-1]}/200 \n')
    f.write(f'***Total cost: {tot_costo}')
    print('PHASE ONE - FINISHED')

# -------------------------------------------------------------------------------
# ******************** PHASE TWO - BEST IMPROVEMENT ********************
# -------------------------------------------------------------------------------
def mejorar_solucion_best_improvement(rutas_nuevas, matriz_distancias, demandas, capacidad):
    mejora_encontrada = True
    costo_total_nuevo = sum(calcular_costo_ruta(ruta, matriz_distancias) for ruta in rutas_nuevas)
    while mejora_encontrada:
        mejora_encontrada = False
        for i in range(len(rutas_nuevas)):
            for j in range(i + 1, len(rutas_nuevas)):
                nueva_ruta_i, nueva_ruta_j = aplicar_best_improvement(rutas_nuevas[i], rutas_nuevas[j], matriz_distancias, demandas, capacidad)
                nuevo_costo_i = calcular_costo_ruta(nueva_ruta_i, matriz_distancias)
                nuevo_costo_j = calcular_costo_ruta(nueva_ruta_j, matriz_distancias)
                if nuevo_costo_i + nuevo_costo_j < calcular_costo_rutas(rutas_nuevas[i], rutas_nuevas[j], matriz_distancias):
                    rutas_nuevas[i], rutas_nuevas[j] = nueva_ruta_i, nueva_ruta_j
                    costo_total_nuevo = sum(calcular_costo_ruta(ruta, matriz_distancias) for ruta in rutas_nuevas)
                    mejora_encontrada = True
    return rutas_nuevas, costo_total_nuevo

def aplicar_best_improvement(ruta_i, ruta_j, matriz_distancias, demandas, capacidad):
    mejor_intercambio_i, mejor_intercambio_j = None, None
    mejor_costo = float('inf')
    for i in range(1, len(ruta_i) - 1):
        for j in range(1, len(ruta_j) - 1):
            nuevo_ruta_i, nuevo_ruta_j = intercambiar_nodos(ruta_i, ruta_j, i, j)
            if verificar_capacidad(nuevo_ruta_i, demandas, capacidad) and verificar_capacidad(nuevo_ruta_j, demandas, capacidad):
                nuevo_costo = calcular_costo_rutas(nuevo_ruta_i, nuevo_ruta_j, matriz_distancias)
                if nuevo_costo < mejor_costo:
                    mejor_costo = nuevo_costo
                    mejor_intercambio_i, mejor_intercambio_j = nuevo_ruta_i, nuevo_ruta_j
    return mejor_intercambio_i, mejor_intercambio_j

def intercambiar_nodos(ruta_i, ruta_j, indice_i, indice_j):
    nueva_ruta_i = ruta_i[:indice_i] + ruta_j[indice_j:]
    nueva_ruta_j = ruta_j[:indice_j] + ruta_i[indice_i:]
    return nueva_ruta_i, nueva_ruta_j

def verificar_capacidad(ruta, demandas, capacidad):
    capacidad_actual = 0
    for nodo in ruta:
        capacidad_actual += demandas[nodo]
        if capacidad_actual > capacidad:
            return False
    return True

def calcular_costo_ruta(ruta, matriz_distancias):
    costo = 0
    for i in range(len(ruta) - 1):
        costo += matriz_distancias[ruta[i]][ruta[i+1]]
    return costo

def calcular_costo_rutas(ruta_i, ruta_j, matriz_distancias):
    costo_i = calcular_costo_ruta(ruta_i, matriz_distancias)
    costo_j = calcular_costo_ruta(ruta_j, matriz_distancias)
    return costo_i + costo_j

# -------------------------------------------------------------------------------
# Extra processing
rutas_mejoradas, costo_total_nuevo = mejorar_solucion_best_improvement(rutas, matriz_distancias, demandas, 35)

# Adding results into the file
with open("Results.txt", "a") as f:
    f.write('\n\n==== PHASE TWO :: RESULTS - BEST IMPROVEMENT ALGORITHM ==== \n')
    for i, ruta in enumerate(rutas_mejoradas, 1):
        costo_ruta = calcular_costo_ruta(ruta, matriz_distancias)
        capacidad_utilizada = sum(demandas[nodo] for nodo in ruta)
        f.write(f'R{i}: {ruta} \n\tCOST: {costo_ruta} -- CAPACITY: {capacidad_utilizada}/200 \n')
    f.write(f'***Total cost: {costo_total_nuevo}')
    print('PHASE TWO - FINISHED')


# -------------------------------------------------------------------------------
# ******************** PHASE THREE - MULTI-START HEURISTIC *********************
# -------------------------------------------------------------------------------
def multi_start_heuristic(instancia_texto, num_inicios, capacidad):
    tic = time.process_time()
    coordenadas, demandas = procesar_instancia(instancia_texto)
    matriz_distancias = calcular_matriz_distancias(coordenadas)
    
    mejor_solucion = None
    mejor_costo = float('inf')
    
    for _ in range(num_inicios):
        rutas, ruta_costo, tot_costo, used_q = vecino_mas_cercano(matriz_distancias, demandas, capacidad)
        rutas_mejoradas, costo_total_nuevo = mejorar_solucion_best_improvement(rutas, matriz_distancias, demandas, capacidad)
        
        if costo_total_nuevo < mejor_costo:
            mejor_solucion = rutas_mejoradas
            mejor_costo = costo_total_nuevo
    
    toc = time.process_time()
    timer = toc - tic
    return mejor_solucion, mejor_costo, coordenadas, demandas, timer


# -------------------------------------------------------------------------------
# Extra processing
num_inicios = 10
capacidad = 35
mejor_solucion, mejor_costo, coordenadas, demandas, timer = multi_start_heuristic(instancia_texto, num_inicios, capacidad)

# Saving results as a new file
with open("Results.txt", "a") as f:
    f.write('\n\n==== MULTI-START HEURISTIC RESULTS ==== \n')
    for i, ruta in enumerate(mejor_solucion, 1):
        costo_ruta = calcular_costo_ruta(ruta, calcular_matriz_distancias(coordenadas))
        capacidad_utilizada = sum(demandas[nodo] for nodo in ruta)
        f.write(f'R{i}: {ruta} \n\tCOST: {costo_ruta} -- CAPACITY: {capacidad_utilizada}/{capacidad} \n')
    f.write(f'*Total cost: {mejor_costo}')
    print(f'MULTI-START HEURISTIC - FINISHED :: {timer} s')

# -------------------------------------------------------------------------------
# Graph
def plot_rutas(coordenadas, rutas):
    plt.figure(figsize=(8, 6))
    coordenadas = np.array(coordenadas)
    plt.scatter(coordenadas[:, 0], coordenadas[:, 1], color='blue', label='Cities')
    for i, ruta in enumerate(rutas):
        x = [coordenadas[nodo][0] for nodo in ruta]
        y = [coordenadas[nodo][1] for nodo in ruta]
        x.append(coordenadas[ruta[0]][0])
        y.append(coordenadas[ruta[0]][1])
        plt.plot(x, y, label=f'R{i+1}')

    plt.legend()
    plt.title('CAPACITATED VEHICLE ROUTING PROBLEM - INSTANCE GRAPH')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    plt.grid(False)
    plt.show()

plot_rutas(coordenadas, mejor_solucion)