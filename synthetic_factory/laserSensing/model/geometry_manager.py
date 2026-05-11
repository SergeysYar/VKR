import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
from shapely.ops import unary_union
from pulp import (
    LpProblem,
    LpVariable,
    LpBinary,
    lpSum,
    LpMinimize,
    value,
    PULP_CBC_CMD,
)


class GeometryManager:
    def __init__(self):
        self.region = None
        self.buildings = []
        self.current_building = []
        self.selected_centers = []

        self._building_tree = None

        # Отладочные данные
        self.last_candidate_centers = []
        self.last_coverage_points = []
        self.last_facade_points = []
        self.last_uncoverable_points = []

    # -----------------------------
    # Базовые операции с геометрией
    # -----------------------------
    def add_region_point(self, point):
        if self.region is None:
            self.region = [point]
        else:
            self.region.append(point)

    def add_building_point(self, point):
        self.current_building.append(point)

    def finish_region(self):
        if self.region is not None and len(self.region) > 2:
            self.region = Polygon(self.region)
            self._rebuild_spatial_index()
            return True
        return False

    def finish_building(self):
        if len(self.current_building) > 2:
            building = Polygon(self.current_building)
            self.buildings.append(building)
            self.current_building = []
            self._rebuild_spatial_index()
            return True
        return False

    def _rebuild_spatial_index(self):
        self._building_tree = STRtree(self.buildings) if self.buildings else None

    def _query_buildings(self, geom):
        """
        Совместимо с версиями shapely, где STRtree.query()
        возвращает либо индексы, либо сами геометрии.
        """
        if self._building_tree is None:
            return []

        result = self._building_tree.query(geom)
        if len(result) == 0:
            return []

        first = result[0]
        if isinstance(first, (int, np.integer)):
            return [self.buildings[int(idx)] for idx in result]

        return list(result)

    def _dedup_points(self, points, digits=6):
        unique = {}
        for item in points:
            if len(item) == 2:
                x, y = item
                key = (round(float(x), digits), round(float(y), digits))
                unique[key] = (float(x), float(y))
            else:
                x, y, nx, ny, kind = item
                key = (
                    round(float(x), digits),
                    round(float(y), digits),
                    round(float(nx), digits),
                    round(float(ny), digits),
                    str(kind),
                )
                unique[key] = (
                    float(x),
                    float(y),
                    float(nx),
                    float(ny),
                    str(kind),
                )
        return list(unique.values())

    def _is_valid_free_point(self, x, y):
        """
        Допустимая точка стояния/покрытия:
        - лежит внутри или на границе участка
        - не лежит внутри/на границе здания
        """
        point = Point(float(x), float(y))

        if self.region is None or not self.region.covers(point):
            return False

        nearby_buildings = self._query_buildings(point)
        return not any(building.covers(point) for building in nearby_buildings)

    def _filter_free_points(self, points):
        result = []
        for x, y in points:
            if self._is_valid_free_point(x, y):
                result.append((float(x), float(y)))
        return self._dedup_points(result)

    def _segment_blocked_by_building(self, start_point, end_point):
        """
        Луч между сканером и точкой не должен проходить через здание.
        """
        line = LineString([start_point, end_point])
        nearby_buildings = self._query_buildings(line)

        for building in nearby_buildings:
            if line.within(building) or line.crosses(building):
                return True
        return False

    # -----------------------------
    # Служебная математика
    # -----------------------------
    @staticmethod
    def _normalize_vector(vx, vy):
        norm = math.hypot(vx, vy)
        if norm == 0:
            return 0.0, 0.0
        return vx / norm, vy / norm

    @staticmethod
    def _polygon_signed_area(coords):
        """
        > 0 -> CCW
        < 0 -> CW
        """
        s = 0.0
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            s += x1 * y2 - x2 * y1
        return 0.5 * s

    def _sample_linestring_points(self, line, step):
        if step <= 0:
            return []

        length = line.length
        if length == 0:
            p = line.interpolate(0)
            return [(float(p.x), float(p.y))]

        distances = np.arange(0.0, length + step, step)
        points = []

        for d in distances:
            d = min(float(d), float(length))
            p = line.interpolate(d)
            points.append((float(p.x), float(p.y)))

        return self._dedup_points(points)

    # -----------------------------
    # Генерация сеток
    # -----------------------------
    def _generate_grid_points(self, step, offsets, validation_func):
        if self.region is None or step <= 0:
            return []

        min_x, min_y, max_x, max_y = self.region.bounds
        points = []

        for offset_x, offset_y in offsets:
            x_coords = np.arange(min_x + offset_x, max_x + step, step)
            y_coords = np.arange(min_y + offset_y, max_y + step, step)

            for x in x_coords:
                for y in y_coords:
                    x = float(x)
                    y = float(y)

                    if x < min_x or x > max_x or y < min_y or y > max_y:
                        continue

                    if validation_func(x, y):
                        points.append((x, y))

        return self._dedup_points(points)

    def _generate_candidate_centers(self, R):
        """
        Грубая сетка кандидатов стояния сканера
        + дополнительные кандидаты около фасадов и вдоль границы.
        """
        center_step = max(1.0, R / 8.0)

        grid_offsets = [
            (0.0, 0.0),
            (center_step / 2.0, 0.0),
            (0.0, center_step / 2.0),
            (center_step / 2.0, center_step / 2.0),
        ]

        grid_candidates = self._generate_grid_points(
            center_step,
            grid_offsets,
            self._is_valid_free_point,
        )

        # Кандидаты вдоль границы участка
        boundary_candidates = []
        if self.region is not None:
            boundary_line = LineString(self.region.exterior.coords)
            boundary_candidates.extend(
                self._sample_linestring_points(boundary_line, max(0.75, center_step / 3.0))
            )
            boundary_candidates.extend(
                [(float(x), float(y)) for x, y in self.region.exterior.coords]
            )
            boundary_candidates = self._filter_free_points(boundary_candidates)

        # Кандидаты вдоль фасадов на нескольких смещениях
        ring_candidates = []
        ring_offsets = [
            0.25,
            0.75,
            1.5,
            3.0,
            max(4.0, center_step / 2.0),
        ]
        ring_step = max(0.5, center_step / 4.0)

        for building in self.buildings:
            for offset in ring_offsets:
                ring = building.buffer(offset, join_style=2).exterior
                ring_candidates.extend(self._sample_linestring_points(ring, ring_step))
                ring_candidates.extend([(float(x), float(y)) for x, y in ring.coords])

        ring_candidates = self._filter_free_points(ring_candidates)

        all_candidates = self._dedup_points(
            grid_candidates + boundary_candidates + ring_candidates
        )

        self.last_candidate_centers = all_candidates
        return all_candidates

    def _generate_free_space_coverage_points(self, R):
        """
        Более плотная сетка точек, которые должны покрываться.
        """
        step = max(0.75, R / 24.0)
        offsets = [
            (0.0, 0.0),
            (step / 2.0, 0.0),
            (0.0, step / 2.0),
            (step / 2.0, step / 2.0),
        ]
        return self._generate_grid_points(step, offsets, self._is_valid_free_point)

    def _generate_boundary_coverage_points(self, R):
        """
        Граница участка должна быть покрыта.
        """
        if self.region is None:
            return []

        step = max(0.5, R / 30.0)
        boundary_line = LineString(self.region.exterior.coords)
        points = self._sample_linestring_points(boundary_line, step)
        points.extend((float(x), float(y)) for x, y in self.region.exterior.coords)

        return self._filter_free_points(points)

    def _generate_facade_points_with_normals(self, R):
        """
        Генерирует точки фасада с нормалью.
        Для каждой точки фасада хранится:
            (x, y, nx, ny, "facade")
        где nx, ny — внешняя нормаль фасада.
        """
        facade_points = []

        if not self.buildings:
            return facade_points

        step = max(0.4, R / 40.0)
        facade_offset = 0.20

        for building in self.buildings:
            coords = list(building.exterior.coords)
            if len(coords) < 2:
                continue

            signed_area = self._polygon_signed_area(coords)
            is_ccw = signed_area > 0

            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]

                segment = LineString([(x1, y1), (x2, y2)])
                seg_len = segment.length
                if seg_len == 0:
                    continue

                # Для CCW внешняя нормаль вправо от направления ребра.
                # Для CW — влево.
                dx = x2 - x1
                dy = y2 - y1

                if is_ccw:
                    nx, ny = dy, -dx
                else:
                    nx, ny = -dy, dx

                nx, ny = self._normalize_vector(nx, ny)

                sample_points = self._sample_linestring_points(segment, step)

                for px, py in sample_points:
                    # Смещаем фасадную контрольную точку немного наружу,
                    # чтобы требовать наблюдение именно с внешней стороны.
                    fx = px + facade_offset * nx
                    fy = py + facade_offset * ny

                    if self._is_valid_free_point(fx, fy):
                        facade_points.append((fx, fy, nx, ny, "facade"))

        facade_points = self._dedup_points(facade_points)
        self.last_facade_points = facade_points
        return facade_points

    def get_coverage_points(self, R_meters=20.0):
        """
        Возвращает список контрольных точек двух типов:
        1) обычные точки пространства: (x, y, 0, 0, "free")
        2) фасадные точки:          (x, y, nx, ny, "facade")
        """
        R = float(R_meters)

        free_points = self._generate_free_space_coverage_points(R)
        boundary_points = self._generate_boundary_coverage_points(R)
        facade_points = self._generate_facade_points_with_normals(R)

        result = []
        for x, y in self._dedup_points(free_points + boundary_points):
            result.append((x, y, 0.0, 0.0, "free"))

        result.extend(facade_points)

        result = self._dedup_points(result)
        self.last_coverage_points = result
        return result

    # -----------------------------
    # Проверка покрытия точки
    # -----------------------------
    def _is_point_covered_from_center(self, center, coverage_point, R):
        """
        coverage_point: (x, y, nx, ny, kind)
        kind in {"free", "facade"}
        """
        cx, cy = center
        px, py, nx, ny, kind = coverage_point

        dx = px - cx
        dy = py - cy

        # 1. Радиус
        if dx * dx + dy * dy > R * R:
            return False

        # 2. Нет пересечения зданий лучом
        if self._segment_blocked_by_building((cx, cy), (px, py)):
            return False

        # 3. Для фасадов нужен правильный угол обзора
        if kind == "facade":
            vx, vy = self._normalize_vector(cx - px, cy - py)
            # Сканер должен находиться примерно со стороны внешней нормали.
            # cos(theta) = n dot view_dir
            dot = nx * vx + ny * vy

            # Порог можно регулировать:
            # 0.2  -> угол до ~78°
            # 0.3  -> до ~72°
            # 0.5  -> до ~60°
            min_dot = 0.25
            if dot < min_dot:
                return False

        return True

    # -----------------------------
    # Оптимизационная модель
    # -----------------------------
    def _create_coverage_dict(self, coverage_points, candidate_centers, R):
        if not coverage_points or not candidate_centers:
            return {}, {}, []

        raw_coverages = []

        cp_xy = np.array([(p[0], p[1]) for p in coverage_points], dtype=float)
        R2 = R * R

        for center in candidate_centers:
            cx, cy = center

            dx = cp_xy[:, 0] - cx
            dy = cp_xy[:, 1] - cy
            near_indices = np.where(dx * dx + dy * dy <= R2)[0]

            covered = []
            for j in near_indices:
                point_info = coverage_points[int(j)]
                if self._is_point_covered_from_center(center, point_info, R):
                    covered.append(int(j))

            if covered:
                raw_coverages.append((center, covered))

        filtered_centers = [item[0] for item in raw_coverages]
        coverage_dict = {i: item[1] for i, item in enumerate(raw_coverages)}

        point_to_centers = {j: [] for j in range(len(coverage_points))}
        for i, pts in coverage_dict.items():
            for j in pts:
                point_to_centers[j].append(i)

        return coverage_dict, point_to_centers, filtered_centers

    def _create_optimization_model(self, point_to_centers, candidate_centers):
        model = LpProblem("LaserScannerCoverage", LpMinimize)

        x_vars = {
            i: LpVariable(f"x_{i}", cat=LpBinary)
            for i in range(len(candidate_centers))
        }

        model += lpSum(x_vars[i] for i in x_vars), "Minimize_scanners"

        # Каждая контрольная точка должна быть покрыта
        for j, centers in point_to_centers.items():
            if centers:
                model += lpSum(x_vars[i] for i in centers) >= 1, f"Cover_point_{j}"

        return model, x_vars

    def optimize_coverage(self, R_meters=20.0):
        if self.region is None:
            print("Регион не задан.")
            return [], 0

        R = float(R_meters)

        coverage_points = self.get_coverage_points(R)
        if not coverage_points:
            print("Нет контрольных точек покрытия.")
            return [], 0

        candidate_centers = self._generate_candidate_centers(R)
        if not candidate_centers:
            print("Нет допустимых точек стояния сканера.")
            return [], 0

        _, point_to_centers, filtered_centers = self._create_coverage_dict(
            coverage_points,
            candidate_centers,
            R,
        )

        uncoverable = [j for j in range(len(coverage_points)) if not point_to_centers[j]]
        self.last_uncoverable_points = [coverage_points[j] for j in uncoverable]

        if uncoverable:
            print(f"Предупреждение: {len(uncoverable)} точек не покрываются.")
            print("Первые непокрываемые точки:")
            for item in self.last_uncoverable_points[:20]:
                x, y, nx, ny, kind = item
                print(f"  {kind}: ({x:.2f}, {y:.2f})")

        if not filtered_centers:
            print("Нет кандидатов, покрывающих хотя бы одну точку.")
            return [], 0

        # Пересчёт по уже отфильтрованным центрам
        _, point_to_centers, filtered_centers = self._create_coverage_dict(
            coverage_points,
            filtered_centers,
            R,
        )

        model, x_vars = self._create_optimization_model(
            point_to_centers,
            filtered_centers,
        )

        solver = PULP_CBC_CMD(msg=False)
        model.solve(solver)

        selected_centers = [
            filtered_centers[i]
            for i in range(len(filtered_centers))
            if value(x_vars[i]) == 1.0
        ]

        self.selected_centers = selected_centers

        print("Выбранные точки стояния сканера:")
        for i, (x, y) in enumerate(selected_centers, start=1):
            print(f"{i}: ({x:.2f}, {y:.2f})")

        objective_value = value(model.objective)
        if objective_value is None:
            objective_value = len(selected_centers)

        return selected_centers, objective_value

    # -----------------------------
    # Старый API / совместимость
    # -----------------------------
    def is_valid_cover(self, center, point):
        """
        Совместимость со старым кодом.
        point может быть либо (x, y), либо расширенной фасадной точкой.
        """
        if len(point) == 2:
            x, y = point
            point_info = (x, y, 0.0, 0.0, "free")
        else:
            point_info = point
        return self._is_point_covered_from_center(center, point_info, R=10**9)

    def calculate_building_coverage(self):
        if self.region is None or not self.buildings:
            return 0

        buildings_union = unary_union(self.buildings)
        intersection = self.region.intersection(buildings_union)
        coverage_area = intersection.area
        total_area = self.region.area
        return (coverage_area / total_area) * 100 if total_area > 0 else 0