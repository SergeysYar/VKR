import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from math import inf
from itertools import combinations


class RobustRoutePlanner:
    def __init__(self, centers, buildings):
        self.centers = centers
        self.buildings = buildings
        self.building_tree = STRtree(buildings)
        self.graph = self._initialize_graph()

    def _initialize_graph(self):
        """Инициализирует граф с центрами в качестве узлов"""
        graph = nx.Graph()  # Используем другое имя переменной
        for idx, pos in enumerate(self.centers):
            graph.add_node(idx, pos=pos, type="center")
        return graph

    def build_network(self):
        """Строит сеть соединений между центрами"""
        # 1. Добавляем прямые соединения
        self._add_direct_connections()

        # 2. Добавляем обходные пути
        self._add_bypass_paths()

        # 3. Оптимизируем сеть
        self._optimize_network()

        return self.graph

    def _add_direct_connections(self):
        """Добавляет прямые связи между центрами"""
        for i, j in combinations(range(len(self.centers)), 2):
            line = LineString([self.centers[i], self.centers[j]])
            if not self._is_obstructed(line):
                dist = Point(self.centers[i]).distance(Point(self.centers[j]))
                self.graph.add_edge(i, j, weight=dist, type="direct")

    def _is_obstructed(self, line):
        """Проверяет пересечение с препятствиями"""
        return any(self.buildings[k].intersects(line)
                   for k in self.building_tree.query(line))

    def _add_bypass_paths(self, num_points=32, offset=0.00015):
        """Добавляет обходные пути для заблокированных соединений"""
        temp_graph = self._create_temp_graph(num_points, offset)

        for i, j in combinations(range(len(self.centers)), 2):
            if not self.graph.has_edge(i, j):
                path = self._find_bypass_path(temp_graph, i, j)
                if path:
                    self.graph.add_edge(i, j, weight=path['length'],
                                        path=path['points'], type="bypass")

    def _create_temp_graph(self, num_points, offset):
        """Создает временный граф с точками обхода"""
        temp_graph = nx.Graph()  # Используем другое имя переменной

        # Добавляем центры
        for i, pos in enumerate(self.centers):
            temp_graph.add_node(f"c_{i}", pos=pos, type="center")

        # Добавляем точки вокруг зданий
        for i, building in enumerate(self.buildings):
            for j, (x, y) in enumerate(building.exterior.coords[:-1]):
                for k in range(num_points):
                    angle = k * (2 * np.pi / num_points)
                    new_x = x + offset * np.cos(angle)  # Явное именование
                    new_y = y + offset * np.sin(angle)  # вместо nx, ny
                    temp_graph.add_node(f"b_{i}_{j}_{k}",
                                        pos=(new_x, new_y),
                                        type="obstacle")

        # Соединяем узлы
        nodes = list(temp_graph.nodes(data=True))
        for (u, u_data), (v, v_data) in combinations(nodes, 2):
            line = LineString([u_data['pos'], v_data['pos']])
            if not self._is_obstructed(line):
                dist = Point(u_data['pos']).distance(Point(v_data['pos']))
                temp_graph.add_edge(u, v, weight=dist)

        return temp_graph

    def _find_bypass_path(self, temp_graph, i, j):
        """Находит обходной путь между центрами"""
        try:
            path = nx.shortest_path(temp_graph, f"c_{i}", f"c_{j}", weight='weight')
            points = [temp_graph.nodes[n]['pos'] for n in path]
            length = nx.shortest_path_length(temp_graph, f"c_{i}", f"c_{j}", weight='weight')
            return {'points': points, 'length': length}
        except nx.NetworkXNoPath:
            return None

    def _optimize_network(self):
        """Оптимизирует сеть соединений"""
        mst = nx.minimum_spanning_tree(self.graph, weight='weight')

        # Добавляем 25% кратчайших связей для надежности
        additional_edges = sorted(self.graph.edges(data=True),
                                  key=lambda x: x[2]['weight'])[:len(self.graph.edges) // 4]

        for u, v, data in additional_edges:
            if not mst.has_edge(u, v):
                mst.add_edge(u, v, **data)

        self.graph = mst

    def find_optimal_route(self):
        """Находит оптимальный маршрут через все центры"""
        if not self.graph.edges():
            self.build_network()

        if not nx.is_connected(self.graph):
            print("Ошибка: не все центры связаны")
            return None

        try:
            path = nx.approximation.traveling_salesman_problem(self.graph, cycle=False)
            route = []

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if 'path' in self.graph.edges[u, v]:
                    route.extend(self.graph.edges[u, v]['path'][:-1])
                else:
                    route.append(self.graph.nodes[u]['pos'])

            route.append(self.graph.nodes[path[-1]]['pos'])
            return route

        except nx.NetworkXError as e:
            print(f"Ошибка построения маршрута: {e}")
            return None