##############################################################################################################################################################
##################################################### VORONOI  ################################################################################################
##############################################################################################################################################################

# Importamos las librerías necesarias
import numpy as np
import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from matplotlib.patches import Circle
from collections import defaultdict

def add_new_point(vor, dists, radius_tot, points, x_limit=(0, 2), y_limit=(0, 2)):
    np.random.seed(seed=0)
    
    while True:
        # Obtengo aquellas posiciones en las que la distancia es mayor que el radio
        posiciones = [i for i in range(len(dists)) if dists[i] > radius_tot[i]]

        # Si no hay posiciones que cumplan con la condición, retornamos None
        if not posiciones:
            return None

        # Elijo una posición al azar entre las que cumplen con la condición
        posicion = random.choice(posiciones)

        # Obtengo los vértices que determinan el diagrama de Voronoi para ese punto
        ridges = np.where(vor.ridge_points == posicion)[0]
        vertex_set = set(np.array(vor.ridge_vertices)[ridges, :].ravel())
        region = [x for x in vor.regions if set(x) == vertex_set][0]
        region = [x for x in region if x != -1]  # Remove outliers
        polygon = vor.vertices[region]

        # Encuentro la distancia con respecto al punto considerado
        point = points[posicion]
        distances = cdist([point], polygon).T
        max_dist_idx = np.argmax(distances)

        # Tomo como punto a considerar el vértice más lejano
        new_point = polygon[max_dist_idx]

        # Verificar si el nuevo punto está dentro de los límites
        if (x_limit[0] < new_point[0] < x_limit[1]) and (y_limit[0] < new_point[1] < y_limit[1]):
            print('El punto {} está dentro de los límites'.format(new_point))
            return new_point
        else:
            print('El punto {} no está dentro de los límites'.format(new_point))

def voronoi_polygons(voronoi, diameter):
    """Generate shapely.geometry.Polygon objects corresponding to the
    regions of a scipy.spatial.Voronoi object, in the order of the
    input points. The polygons for the infinite regions are large
    enough that all points within a distance 'diameter' of a Voronoi
    vertex are contained in one of the infinite polygons.

    """
    ## Se calcula el centroide (Esto nos servirá luego para determinar la dirección de los bordes infinitos)
    ## De manera general queremos que apunten del centro hacia afuera (y no al revés) para que se expandan hacia los boundaries
    centroid = voronoi.points.mean(axis=0)

    # Mapping from (input point index, Voronoi point index) to list of
    # unit vectors in the directions of the infinite ridges starting
    # at the Voronoi point and neighbouring the input point.

    ridge_direction = defaultdict(list) ## Esto es un diccionario con los puntos y los vectores asociados. Ejem [(0,1):[-1,0],..]
    ## En el caso anterior significaría que la direccion del borde infinito con respecto al punto 0 y el vertice 0 sería [-1,0] (partiendo desde el vertice 0)

    ## Ridge points contiene entre que pares de puntos existe un borde. Ejemplo: [0,3] -> Existe un vertice entre el point 0 y el point 3
    # Ridge vertices contiene entre que pares de vertices existe un borde. Ejemplo: [0,1] -> Existe un vertice entre el vertice 0 y el vertice 1
    # Si pone -1, significa que el vertice es infinito. Eje: [0,-1] -> Hay un borde infinito que parte del vertice 0
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv) ## Se ordena por cuestion de comodidad para buscar los bordes infinitos siempre en la primera posicion (u)
        if u == -1:
            ## Se construye el vector que va desde el punto p al punto q
            t = voronoi.points[q] - voronoi.points[p] 

            ## Se extrae el vector normal al pq
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)

            ## Se calcula el punto medio entre p y q
            midpoint = voronoi.points[[p, q]].mean(axis=0)

            ## Se comprueba si la direccion del vector es la correcta o debe ser la opuesta
            direction = np.sign(np.dot(midpoint - centroid, n)) * n

            ## Se añaden las direccion de los bordes infinitos
            ridge_direction[p, v].append(direction) ### Direccion entre el punto p y el vertice v -> Ejemplo: (0,1):[-1,0] -> Direccion entre el punto 0 y el vertice 1 es [-1,0]
            ridge_direction[q, v].append(direction) ### Direccion entre el punto q y el vertice v
        #print(ridge_direction)
    
    """si el producto escalar es positivo, significa que la dirección debe ser igual a n, y si es negativo, la dirección debe ser igual a -n.
Esto viene de la formula del producto escalar ya que A.B = |A||B|cos(theta). Si el angulo es menor a 90 grados, el coseno es positivo, y si es mayor a 90 grados, el coseno es negativo.
    """

    ## Point region contiene el indice de la region de Voronoi a la que pertenece cada punto
    ## Ejemplo: point_region[0] = 0 -> El punto 0 pertenece a la region 0
    for i, r in enumerate(voronoi.point_region):
        ## Extraemos la region de Voronoi a la que pertenece el punto i
        region = voronoi.regions[r]
        ## En caso de que -1 no esté en la region, significa que es una region finita luego ya habriamos acabado
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        ##La idea de tomar el modulo es porque en caso de que el vertice de Voronoi sea el primero o el ultimo,
        ##el indice anterior es el ultimo del ciclo o el primero del ciclo respectivamente.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]

        # Length of ridges needed for the extra edge to lie at least
        # 'diameter' away from all Voronoi vertices.
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)

        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def add_new_point_polygon(vor, dists, radius_tot,boundary):
    """Funcion corregida para poder usar como vertices los limites del poligono donde corta con los voronoi set

    Args:
        vor (_type_): Voronoi diagram from scipy
        dists (_type_): List of distances from each point to the furthest vertex
        radius_tot (_type_): List of radius from each point 
        boundary (_type_): Polygon that defines the boundary of the space

    Returns:
        new_point: New point to be considered
    """
    np.random.seed(seed=0)
    while True:
        # Obtengo aquellas posiciones en las que la distancia es mayor que el radio
        posiciones = [i for i in range(len(dists)) if dists[i] > radius_tot[i]]
        # Si no hay posiciones que cumplan con la condición, retornamos None y terminamos
        if not posiciones:
            return None
        # Elijo una posición al azar entre las que cumplen con la condición
        posicion = random.choice(posiciones)
        point_inside = Point(vor.points[posicion])
        diameter = np.linalg.norm(boundary.ptp(axis=0))

        ## Cargamos los poligonos que definen cada voronoi set
        polygons_tot = voronoi_polygons(vor, diameter)
        polygons = list(polygons_tot)
        p = polygons[posicion]
        boundary_polygon = Polygon(boundary)
        polygon = p.intersection(boundary_polygon)
        #vertices = np.array(list(polygon.exterior.coords))
        furthest_vertex = find_furthest_vertex(polygon, point_inside)
        new_point = furthest_vertex

        return new_point

def add_new_point_polygon_optimized(vor, dists, radius_tot, boundary):
    """Funcion corregida para calcular el vértice más lejano que cubra el menor número de puntos originales.

    Args:
        vor (Voronoi): Voronoi diagram from scipy
        dists (list): List of distances from each point to the furthest vertex
        radius_tot (list): List of radius from each point
        boundary (Polygon): Polygon that defines the boundary of the space

    Returns:
        new_point: New point to be considered
    """
    np.random.seed(seed=0)
    min_covered_points = float('inf') ## Es una forma de asegurarnos
    selected_vertex = None

    for i, dist in enumerate(dists):
        if dist > radius_tot[i]:
            point_inside = Point(vor.points[i])
            diameter = np.linalg.norm(boundary.ptp(axis=0))

            # Cargamos los polígonos que definen cada voronoi set
            polygons_tot = voronoi_polygons(vor, diameter)
            polygons = list(polygons_tot)
            p = polygons[i]
            boundary_polygon = Polygon(boundary)
            polygon = p.intersection(boundary_polygon)

            furthest_vertex = find_furthest_vertex(polygon, point_inside)

            # Calcular cuántos puntos originales están cubiertos por este vértice
            ## Esto es equivalente a saber cuantos de los puntos originales cubren el vertice
            covered_points = sum(1 for j in range(len(vor.points)) if np.linalg.norm(vor.points[j] - furthest_vertex) < radius_tot[j])
            # Actualizar si este vértice cubre menos puntos originales
            if covered_points < min_covered_points:
                min_covered_points = covered_points
                selected_vertex = furthest_vertex

    return selected_vertex

def find_furthest_vertex(polygon, point):
    distances_to_vertices = [point.distance(Point(x, y)) for x, y in polygon.exterior.coords]
    furthest_vertex_index = distances_to_vertices.index(max(distances_to_vertices))
    return polygon.exterior.coords[furthest_vertex_index]

def plot_voronoi_with_furthest_vertices(vor, boundary, radius_tot):
    x, y = boundary.T
    plt.xlim(round(x.min() - 1), round(x.max() + 1))
    plt.ylim(round(y.min() - 1), round(y.max() + 1))
    plt.plot(*points.T, 'b.')

    diameter = np.linalg.norm(boundary.ptp(axis=0))
    boundary_polygon = Polygon(boundary)

    
    max_distances = []
    for i, p in enumerate(voronoi_polygons(vor, diameter)):
        point_inside = Point(vor.points[i]) 
        polygon = p.intersection(boundary_polygon)

        x, y = zip(*polygon.exterior.coords)
        plt.plot(x, y, 'r-')

        furthest_vertex = find_furthest_vertex(polygon, point_inside)
        plt.scatter(furthest_vertex[0], furthest_vertex[1], color='green', label="Furthest Vertex")

        # Dibuja la línea desde el punto al vértice más lejano
        plt.plot([point_inside.x, furthest_vertex[0]], [point_inside.y, furthest_vertex[1]], linestyle='--', color='purple', label="Distance Line")
        
        # Dibuja un círculo alrededor del punto con el radio correspondiente
        circle_fill = Circle((point_inside.x, point_inside.y), radius_tot[i], color='red', alpha=0.25)
        plt.gca().add_patch(circle_fill)

        # Calcula la máxima distancia
        max_distance = max([point_inside.distance(Point(x, y)) for x, y in polygon.exterior.coords])
        max_distances.append(max_distance)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Voronoi Diagram with Furthest Vertices, Distance Lines, and Circles')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    return max_distances


def plot_voronoi_with_all_vertices(vor, boundary, dict_radios):
    x, y = boundary.T
    plt.xlim(round(x.min() - 1), round(x.max() + 1))
    plt.ylim(round(y.min() - 1), round(y.max() + 1))
    plt.plot(*vor.points.T, 'b.')

    diameter = np.linalg.norm(boundary.ptp(axis=0))
    boundary_polygon = Polygon(boundary)

    radius_tot = [v[0] for k,v in dict_radios.items()]
    signo = [v[1] for k,v in dict_radios.items()]

    max_distances = []
    for i, p in enumerate(voronoi_polygons(vor, diameter)):
        point_inside = Point(vor.points[i]) 
        polygon = p.intersection(boundary_polygon)

        x, y = zip(*polygon.exterior.coords)
        plt.plot(x, y, 'r-')

        vertices = np.array(list(polygon.exterior.coords))
        plt.scatter(vertices[:, 0], vertices[:, 1], color='green', label="Vertices")

        furthest_vertex = find_furthest_vertex(polygon, point_inside)
        plt.scatter(furthest_vertex[0], furthest_vertex[1], color='orange', label="Furthest Vertex")

        # Dibuja la línea desde el punto al vértice más lejano
        plt.plot([point_inside.x, furthest_vertex[0]], [point_inside.y, furthest_vertex[1]], linestyle='--', color='purple', label="Distance Line")

        # Dibuja un círculo alrededor del punto con el radio correspondiente (rojo positivo, azul negativo)
        if signo[i]>0:
            circle_fill = Circle((point_inside.x, point_inside.y), radius_tot[i], color='red', alpha=0.25)
            plt.gca().add_patch(circle_fill)
        else:
            circle_fill = Circle((point_inside.x, point_inside.y), radius_tot[i], color='blue', alpha=0.25)
            plt.gca().add_patch(circle_fill)

        # Calcula la máxima distancia
        max_distance = max([point_inside.distance(Point(x, y)) for x, y in polygon.exterior.coords])
        max_distances.append(max_distance)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Voronoi Diagram with All Vertices, Circles, and Distance Lines')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    return max_distances

# Uso de la función:
# new_point = add_new_point(vor, dists, radius_tot, points)

def generate_voronoi_diagram(vor,radius_tot,verbose=False,save_gif=False,frame=0):
    np.random.seed(seed=0)
    points = vor.points
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    dists = []

    max_distances = []  # To track the maximum distance for each point
    for i, point in enumerate(points):

        # Update the maximum distance
        distances = cdist([point], vor.vertices).T
        max_dist = np.max(distances)
        max_distances.append(max_dist)

        # Plot a circle around the point with the chosen radius
        circle = Circle((point[0], point[1]), radius_tot[i], color='blue', fill=False)
        ax.add_patch(circle)

        # Fill the interior of the circle in red with alpha 0.25
        circle_fill = Circle((point[0], point[1]), radius_tot[i], color='red', alpha=0.25)
        ax.add_patch(circle_fill)

        # get nearby vertices
        ridges = np.where(vor.ridge_points == i)[0]
        vertex_set = set(np.array(vor.ridge_vertices)[ridges, :].ravel())
        region = [x for x in vor.regions if set(x) == vertex_set][0]
        region = [x for x in region if x != -1]  # remove outliers
        polygon = vor.vertices[region]
        if len(polygon) < 1:
            continue

        ### En polygon tengo puestos los vértices que determinan cada region

        # calc distance of every vertex to the initial point
        distances = cdist([point], polygon).T
        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx][0]  # Extract the scalar value
        dists.append(max_dist)

        # just for visuals (Con esto estoy dibujando las lineas que unen los puntos originales con el más lejano)
        xvals = [point[0], polygon[max_dist_idx][0]]
        yvals = [point[1], polygon[max_dist_idx][1]]
        plt.plot(xvals, yvals, 'r-')
        #plt.xlim(-0.2,2.5)
        #plt.ylim(-0.2,2.5)

        if verbose==True:
            # Print distance and radius information
            print(f'Punto {point} distancia máxima: {max_dist}, radio: {radius_tot[i]}')
        if save_gif:
            plt.savefig(f'frame_{frame}.png')



    # Check if all radii are greater than their respective distances
    if all(radius > dist for radius, dist in zip(radius_tot, dists)):
        print("El espacio está relleno.")
    else:
        print("El espacio no está relleno.")
        
    if save_gif:
        plt.close()
    else:
        plt.show()
    return dists, max_distances, radius_tot

def verifica_relleno(intervalos, intervalo_deseado):
    # Ordenar los intervalos por valor de inicio
    intervalos.sort(key=lambda x: x[0])
    
    inicio_deseado, fin_deseado = intervalo_deseado
    
    inicio_actual = inicio_deseado
    fin_actual = inicio_deseado
    
    for intervalo in intervalos:
        if intervalo[0] <= fin_actual:
            # El intervalo se superpone con el intervalo actual
            fin_actual = max(fin_actual, intervalo[1])
        else:
            # El intervalo no se superpone, por lo que no hay relleno completo
            return False
    
    return fin_actual >= fin_deseado


def plot_intervalos(dict_intervalos,intervalo_deseado):
    inicio_deseado, fin_deseado = intervalo_deseado
    intervalos = [i[:2] for i in list(dict_intervalos.values())]
    signo = [i[2] for i in list(dict_intervalos.values())]
    
    # Crear una figura y un eje
    fig, ax = plt.subplots(figsize=(7, 2))
    ax.fill_betweenx(y=[0, 1], x1=inicio_deseado, x2=fin_deseado, color='white')
    # Dibujar la parte cubierta en azul
    for i,intervalo in enumerate(intervalos):
        inicio, fin = intervalo
        signo_i = signo[i]
        if signo_i>0:
            ax.fill_betweenx(y=[0, 1], x1=inicio, x2=fin, color='red',alpha=0.5)
        else:
            ax.fill_betweenx(y=[0, 1], x1=inicio, x2=fin, color='blue',alpha=0.5)
        plt.vlines(inicio, 0, 1, colors='k', linestyles='solid')
        plt.vlines(fin, 0, 1, colors='k', linestyles='solid')
        plt.vlines((inicio+fin)/2, 0, 1, colors='white', linestyles='dashed')
    
    
    
    # Configurar el eje y mostrar el gráfico
    ax.set_xlim(inicio_deseado, fin_deseado)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Eje X')
    ax.set_yticks([])
    plt.ylim(0, 1)

    plt.show()


def calcular_intervalos(x_input, mlp_positive_model, global_lipschitz_constant):
    intervalos = []
    dict_intervalos = {}
    x_reentrenamiento = []

    for x in x_input:
        x_copy = x.clone().detach().requires_grad_(True)
        output = mlp_positive_model(x_copy)
        output.backward()
        derivative = x_copy.grad
        if derivative.item() > 0:
            h = derivative.item() / global_lipschitz_constant.item()
            intervalos.append([x_copy.item() - h, x_copy.item() + h])
            dict_intervalos[x_copy.item()] = [x_copy.item() - h, x_copy.item() + h,1]
        else:
            x_reentrenamiento.append(x_copy.item())
            h = -(derivative.item() / global_lipschitz_constant.item()) ## Cambiamos el signo para que el intervalo sea positivo
            intervalos.append([x_copy.item()-h, x_copy.item()+h])
            dict_intervalos[x_copy.item()] = [x_copy.item()-h, x_copy.item()+h,-1]
        x_copy.grad.zero_()

    return intervalos, dict_intervalos,x_reentrenamiento



def genera_punto_medio(dict_intervalos, intervalo_deseado):
    stop = False
    puntos_medio_seguimiento = {}
    while not stop:
        ## Vector de puntos
        puntos = list(dict_intervalos.keys())
        ## Añadimos el inicio y el final del intervalo
        puntos.append(intervalo_deseado[0])
        puntos.append(intervalo_deseado[1])
        ## Nos quedamos solo con una repeticion por punto
        puntos = list(set(puntos))
        ## Ordenamos los puntos
        puntos.sort()
        ## Generamos el vector de puntos medios
        puntos_medios = []
        for i in range(len(puntos)-1):
            puntos_medios.append((puntos[i]+puntos[i+1])/2)
        ## Buscamos primero un punto medio solo
        for punto_medio in puntos_medios:
            ## Comprobamos si el punto medio está en algún intervalo 
            cuenta = 0
            intervalos = [i[:2] for i in list(dict_intervalos.values())]
            for intervalo in intervalos:
                if punto_medio >= intervalo[0] and punto_medio <= intervalo[1]:
                    cuenta+=1
            puntos_medio_seguimiento[punto_medio] = cuenta
        ## Si existe algun punto medio que no esté en ningún intervalo, lo seleccionamos
        if 0 in list(puntos_medio_seguimiento.values()):
            extremo_sampleado = random.choice([p for p in puntos_medio_seguimiento.keys() if puntos_medio_seguimiento[p] == 0])
            stop = True
            print('El punto {} no está en ningún intervalo'.format(extremo_sampleado))
            return extremo_sampleado
        ## Si no escogemos alguno punto que este en un intervalo
        else:
            ## Escogemos aquel que este cubierto por el menor numero de intervalos
            extremo_sampleado = sorted(puntos_medio_seguimiento.items(), key=lambda x: x[1])[0][0]
            #extremo_sampleado = random.choice([p for p in puntos_medio_seguimiento.keys() if puntos_medio_seguimiento[p] == 1])
            stop = True
            print('El punto {} está en {} intervalos'.format(extremo_sampleado,puntos_medio_seguimiento[extremo_sampleado]))
            return extremo_sampleado