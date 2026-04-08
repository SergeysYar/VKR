import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from scipy.spatial import Delaunay
import warnings

warnings.filterwarnings('ignore')


class ModelFeatureExtractor:
    """
    Извлечение признаков из обученной модели для классификации геометрии.
    """

    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model = None
        self.feature_dim = 256
        self._load_model()

    def _load_model(self):
        """Загрузка модели из чекпоинта."""
        print(f"\nЗагрузка модели из {self.ckpt_path}...")
        try:
            checkpoint = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)

            # Создаем простую модель для извлечения признаков
            self.model = SimpleFeatureExtractor()

            # Загружаем веса
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Фильтруем только нужные ключи
                filtered_dict = {}
                for k, v in state_dict.items():
                    if 'encoder' in k or 'features' in k or 'conv' in k:
                        new_k = k.replace('model.', '').replace('encoder.', '')
                        filtered_dict[new_k] = v

                try:
                    self.model.load_state_dict(filtered_dict, strict=False)
                    print("  Веса загружены")
                except Exception as e:
                    print(f"  Ошибка загрузки весов: {e}")

            self.model.eval()
            print("  Модель готова к использованию")

        except Exception as e:
            print(f"  Не удалось загрузить модель: {e}")
            self.model = None

    def extract_features(self, points):
        """
        Извлечение признаков из облака точек.
        Возвращает вектор признаков размерности feature_dim.
        """
        if self.model is None:
            return None

        if len(points) < 10:
            return None

        # Подготовка точек
        if len(points) > 1024:
            idx = np.random.choice(len(points), 1024, replace=False)
            points = points[idx]
        elif len(points) < 1024:
            idx = np.random.choice(len(points), 1024, replace=True)
            points = points[idx]

        # Нормализация
        center = np.mean(points, axis=0)
        points = points - center
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        # Преобразуем в тензор
        points_tensor = torch.FloatTensor(points).unsqueeze(0)

        # Извлекаем признаки
        with torch.no_grad():
            try:
                features = self.model(points_tensor)
                return features.numpy().flatten()
            except Exception as e:
                print(f"    Ошибка извлечения признаков: {e}")
                return None

    def get_geometry_hint(self, features):
        """
        Получение подсказки о геометрии на основе признаков.
        Использует простую классификацию по паттернам в признаках.
        """
        if features is None:
            return None

        # Анализируем паттерны в признаках
        # Высокая энергия в первых компонентах = линейность
        energy_distribution = np.abs(features[:50]) / (np.sum(np.abs(features)) + 1e-10)

        if np.sum(energy_distribution[:10]) > 0.5:
            return 'linear'  # трубчатые

        if np.sum(energy_distribution[10:20]) > 0.5:
            return 'planar'  # плоские

        if np.var(features) < 0.1:
            return 'uniform'  # равномерные (объемные)

        return 'complex'  # сложные


class SimpleFeatureExtractor(nn.Module):
    """
    Простая сеть для извлечения признаков из облака точек.
    """

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=256):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, N, 3]
        x = self.mlp1(x)  # [B, N, 128]
        x = self.mlp2(x)  # [B, N, hidden_dim]

        x = x.transpose(1, 2)  # [B, hidden_dim, N]
        x = self.pool(x)  # [B, hidden_dim, 1]
        x = x.squeeze(-1)  # [B, hidden_dim]

        x = self.fc(x)  # [B, output_dim]

        return x


class GeometryClassifier:
    """
    Классифицирует геометрию класса используя:
    - геометрические признаки (PCA)
    - признаки из нейросети
    """

    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor

    def classify(self, points):
        """
        Определяет тип геометрии:
        - 'flat' - плоский (ручки, пластины)
        - 'tubular' - трубчатый
        - 'solid' - объемный
        - 'complex' - сложный
        """
        if len(points) < 50:
            return 'small'

        # 1. Геометрические признаки (PCA)
        center = np.mean(points, axis=0)
        centered = points - center

        pca = PCA(n_components=3)
        pca.fit(centered)

        ev = pca.explained_variance_
        ev_norm = ev / (ev.sum() + 1e-10)

        planarity = (ev_norm[1] - ev_norm[2]) / ev_norm[0] if ev_norm[0] > 0 else 0
        linearity = (ev_norm[0] - ev_norm[1]) / ev_norm[0] if ev_norm[0] > 0 else 0

        thickness = np.sqrt(ev[2])
        width = np.sqrt(ev[0] + ev[1])
        thin_ratio = thickness / width if width > 0 else 1

        bbox = points.max(axis=0) - points.min(axis=0)
        bbox_ratio = max(bbox) / (min(bbox) + 1e-10)

        print(f"    PCA: plan={planarity:.2f}, lin={linearity:.2f}, thin={thin_ratio:.3f}, bbox={bbox_ratio:.2f}")

        # 2. Признаки из нейросети
        nn_hint = None
        if self.feature_extractor:
            features = self.feature_extractor.extract_features(points)
            nn_hint = self.feature_extractor.get_geometry_hint(features)
            if nn_hint:
                print(f"    NN hint: {nn_hint}")

        # 3. Комбинированное решение
        # Плоские объекты
        if planarity > 0.5 and thin_ratio < 0.3:
            if nn_hint in [None, 'planar']:
                return 'flat'

        # Трубчатые
        if linearity > 0.5 and bbox_ratio > 2:
            if nn_hint in [None, 'linear']:
                return 'tubular'

        # Объемные
        if ev_norm[0] < 0.6 and ev_norm[1] < 0.6 and ev_norm[2] < 0.6:
            if nn_hint in [None, 'uniform']:
                return 'solid'

        # Если нейросеть уверена
        if nn_hint == 'planar':
            return 'flat'
        if nn_hint == 'linear':
            return 'tubular'
        if nn_hint == 'uniform':
            return 'solid'

        return 'complex'


class FlatReconstructor:
    """Реконструкция плоских объектов."""

    @staticmethod
    def reconstruct(points):
        if len(points) < 4:
            return None

        pca = PCA(n_components=3)
        pca.fit(points)

        normal = pca.components_[2]
        center = np.mean(points, axis=0)
        centered = points - center
        projected = centered - np.outer(np.dot(centered, normal), normal)

        basis1 = pca.components_[0]
        basis2 = pca.components_[1]

        points_2d = np.column_stack([
            np.dot(projected, basis1),
            np.dot(projected, basis2)
        ])

        tri = Delaunay(points_2d)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()

        return mesh


class TubularReconstructor:
    """Реконструкция трубчатых объектов."""

    @staticmethod
    def reconstruct(points):
        if len(points) < 100:
            return None

        pca = PCA(n_components=3)
        pca.fit(points)
        axis = pca.components_[0]
        center = np.mean(points, axis=0)

        centered = points - center
        proj = np.dot(centered, axis)
        radii = np.linalg.norm(centered - np.outer(proj, axis), axis=1)

        if len(radii) > 20:
            radii_scaled = (radii - np.mean(radii)) / (np.std(radii) + 1e-10)
            clustering = DBSCAN(eps=0.3, min_samples=10).fit(radii_scaled.reshape(-1, 1))
            labels = clustering.labels_

            layers = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in layers:
                    layers[label] = []
                layers[label].append(points[i])

            layer_meshes = []
            for layer_points in layers.values():
                layer_array = np.array(layer_points)
                if len(layer_array) >= 10:
                    mesh = TubularReconstructor._connect_layer(layer_array)
                    if mesh:
                        layer_meshes.append(mesh)

            if layer_meshes:
                combined = o3d.geometry.TriangleMesh()
                for mesh in layer_meshes:
                    combined += mesh
                combined.remove_duplicated_vertices()
                combined.remove_duplicated_triangles()
                return combined

        return TubularReconstructor._connect_layer(points)

    @staticmethod
    def _connect_layer(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        try:
            distances = np.asarray(pcd.compute_nearest_neighbor_distance())
            mean_dist = np.mean(distances)
        except:
            mean_dist = 1.0

        for alpha_factor in [1.0, 1.2, 1.5]:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, mean_dist * alpha_factor
                )
                if len(mesh.triangles) > 0:
                    mesh.remove_degenerate_triangles()
                    mesh.remove_unreferenced_vertices()
                    return mesh
            except:
                continue

        return None


class SolidReconstructor:
    """Реконструкция объемных объектов."""

    @staticmethod
    def reconstruct(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        try:
            distances = np.asarray(pcd.compute_nearest_neighbor_distance())
            mean_dist = np.mean(distances)
        except:
            mean_dist = 1.0

        for alpha_factor in [1.0, 1.2, 1.5, 2.0]:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, mean_dist * alpha_factor
                )
                if len(mesh.triangles) > 0:
                    mesh.remove_degenerate_triangles()
                    mesh.remove_unreferenced_vertices()
                    return mesh
            except:
                continue

        return None


class ComplexReconstructor:
    """Реконструкция сложных объектов."""

    @staticmethod
    def reconstruct(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        try:
            distances = np.asarray(pcd.compute_nearest_neighbor_distance())
            mean_dist = np.mean(distances)
        except:
            mean_dist = 1.0

        best_mesh = None
        best_triangles = 0

        for alpha_factor in [1.0, 1.2, 1.5, 2.0]:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, mean_dist * alpha_factor
                )
                if len(mesh.triangles) > best_triangles:
                    best_mesh = mesh
                    best_triangles = len(mesh.triangles)
            except:
                continue

        try:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
            radii = [mean_dist * 0.8, mean_dist * 1.0, mean_dist * 1.2]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            if len(mesh.triangles) > best_triangles:
                best_mesh = mesh
                best_triangles = len(mesh.triangles)
        except:
            pass

        if best_mesh:
            best_mesh.remove_degenerate_triangles()
            best_mesh.remove_unreferenced_vertices()

        return best_mesh


class SmartReconstructor:
    """
    Умный реконструктор с использованием нейросетевых признаков.
    """

    def __init__(self, input_path, ckpt_path=None, output_dir="./results", min_points=50):
        self.input_path = input_path
        self.ckpt_path = ckpt_path
        self.output_dir = Path(output_dir)
        self.min_points = min_points
        self.output_dir.mkdir(exist_ok=True)

        # Загружаем данные
        self.points, self.labels = self._load_ply()

        # Загружаем экстрактор признаков из модели
        self.feature_extractor = None
        if ckpt_path:
            self.feature_extractor = ModelFeatureExtractor(ckpt_path)

        # Группируем по классам
        self.classes = self._group_by_class()

        self.classifier = GeometryClassifier(self.feature_extractor)

        print(f"\nЗагружено: {len(self.points)} точек, {len(self.classes)} классов")

    def _load_ply(self):
        points = []
        labels = []

        with open(self.input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start = 0
        for i, line in enumerate(lines):
            if line.startswith('end_header'):
                start = i + 1
                break

        for line in lines[start:]:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    label = int(float(parts[3]))
                    points.append([x, y, z])
                    labels.append(label)

        return np.array(points), np.array(labels)

    def _group_by_class(self):
        classes = {}
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            mask = self.labels == label
            class_points = self.points[mask]
            label_int = int(label)

            if len(class_points) >= self.min_points:
                classes[label_int] = class_points
                print(f"Класс {label_int}: {len(class_points)} точек")

        return classes

    def reconstruct_class(self, class_id, points):
        """Реконструкция одного класса."""
        print(f"\n{'─' * 50}")
        print(f"КЛАСС {class_id} ({len(points)} точек)")

        # Классифицируем геометрию (с использованием нейросети)
        geom_type = self.classifier.classify(points)
        print(f"  Тип: {geom_type}")

        # Выбираем метод
        mesh = None
        if geom_type == 'flat':
            print(f"  Метод: Flat")
            mesh = FlatReconstructor.reconstruct(points)
        elif geom_type == 'tubular':
            print(f"  Метод: Tubular")
            mesh = TubularReconstructor.reconstruct(points)
        elif geom_type == 'solid':
            print(f"  Метод: Solid")
            mesh = SolidReconstructor.reconstruct(points)
        else:
            print(f"  Метод: Complex")
            mesh = ComplexReconstructor.reconstruct(points)

        if mesh and len(mesh.triangles) > 0:
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()

            print(f"  ✓ Создано: {len(mesh.triangles)} треугольников")

            colors = {
                'flat': [0.9, 0.7, 0.1],
                'tubular': [0.1, 0.7, 0.9],
                'solid': [0.1, 0.9, 0.1],
                'complex': [0.9, 0.1, 0.7]
            }
            color = colors.get(geom_type, [0.5, 0.5, 0.5])
            mesh.paint_uniform_color(color)

            return {
                'class_id': class_id,
                'type': geom_type,
                'mesh': mesh,
                'triangles': len(mesh.triangles)
            }
        else:
            print(f"  ✗ Не удалось")
            return None

    def run(self):
        print("\n" + "=" * 60)
        print("НАЧАЛО РЕКОНСТРУКЦИИ")
        print("=" * 60)

        self.meshes = {}
        results = []

        for class_id, points in self.classes.items():
            result = self.reconstruct_class(class_id, points)
            if result:
                self.meshes[class_id] = result['mesh']
                results.append(result)

                path = self.output_dir / f"class_{class_id:02d}_{result['type']}.ply"
                o3d.io.write_triangle_mesh(str(path), result['mesh'])
                print(f"  Сохранено: {path.name}")

        if self.meshes:
            composite = o3d.geometry.TriangleMesh()
            for mesh in self.meshes.values():
                composite += mesh
            composite.remove_duplicated_vertices()
            composite.remove_duplicated_triangles()
            path = self.output_dir / "complete_model.ply"
            o3d.io.write_triangle_mesh(str(path), composite)
            print(f"\nСохранена полная модель: complete_model.ply")

        self._save_report(results)
        self._visualize_all()

        print(f"\nГотово! Результаты в {self.output_dir}")

    def _save_report(self, results):
        path = self.output_dir / "report.txt"
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("ОТЧЕТ О РЕКОНСТРУКЦИИ\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"Класс {r['class_id']}:\n")
                f.write(f"  Тип: {r['type']}\n")
                f.write(f"  Треугольников: {r['triangles']}\n\n")

    def _visualize_all(self):
        if not self.meshes:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        geometries = [pcd] + list(self.meshes.values())
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Реконструкция с нейросетью",
            width=1280,
            height=720
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Реконструкция с нейросетевыми признаками')
    parser.add_argument('input', help='PLY файл с метками')
    parser.add_argument('--ckpt', help='Чекпоинт модели (опционально)')
    parser.add_argument('--output', '-o', default='./results')
    parser.add_argument('--min-points', type=int, default=50)

    args = parser.parse_args()

    reconstructor = SmartReconstructor(
        args.input,
        ckpt_path=args.ckpt,
        output_dir=args.output,
        min_points=args.min_points
    )

    reconstructor.run()


if __name__ == "__main__":
    main()