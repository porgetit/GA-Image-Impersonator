from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
from dataclasses import dataclass
from PIL import Image


# Carga y preprocesamiento
def load_image(
    path: str,
    to_shape: Tuple[int, int] | None = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Carga una imagen RGB como np.ndarray (H, W, C). Si to_shape se proporciona, redimensiona a (W, H).
    Si normalize es True, escala píxeles a [0, 1] en float32.
    """
    img = Image.open(path).convert("RGB")
    if to_shape is not None:
        w, h = to_shape[1], to_shape[0] # PIL usa (W, H)
        img = img.resize((w, h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    if normalize:
        arr /= 255.0
        
    return arr # (H, W, 3)

# Genoma (modo 'píxeles')
@dataclass
class GenomeSpec:
    H: int
    W: int
    C: int = 3  # Canales RGB
    low: float = 0.0
    high: float = 1.0
    
    @property
    def gene_dim(self) -> int:
        return self.H * self.W * self.C
    
def make_genome_spec(
    H: int,
    W: int,
    C: int,
    low: float,
    high: float
) -> GenomeSpec:
    """
    Crea y devuelve un GenomeSpec.
    """
    return GenomeSpec(H=H, W=W, C=C, low=low, high=high)

def decode(genes: list[float], spec: GenomeSpec) -> np.ndarray:
    """
    Decodifica una lista de genes a una imagen np.ndarray (H, W, C) según el GenomeSpec.
    """
    arr = np.array(genes, dtype=np.float32).reshape((spec.H, spec.W, spec.C))
    return arr

# Estadísticas objetivo
@dataclass
class TargetStats:
    mu_globabl: np.ndarray  # Media global por canal (C, )
    mu_rows: np.ndarray    # Media por fila y canal (H, C)
    mu_cols: np.ndarray    # Media por columna y canal (W, C)
    
def prepare_target_stats(I: np.ndarray) -> TargetStats:
    """
    Precalcula media global, por filas y por columnas de la imagen objetivo I (H, W, C).
    """
    H, W, C = I.shape
    mu_global = I.reshape(-1, C).mean(axis=0)          # (C, )
    mu_rows = I.mean(axis=1)                           # (H, C) <-- OJO: axis=1 promedia sobre columnas? verificar
    mu_cols = I.mean(axis=0)                           # (W, C)
    return TargetStats(mu_globabl=mu_global, mu_rows=mu_rows, mu_cols=mu_cols)

# Fitness combinada (DEAP Eval)
def _candidate_stats(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula estadísticas de la imagen candidata img (H, W, C):
    - mu_global (C, )
    - mu_rows (H, C)
    - mu_cols (W, C)
    """
    H, W, C = img.shape
    mu_global = img.reshape(-1, C).mean(axis=0)        # (C, )
    mu_rows = img.mean(axis=1)                         # (H, C)
    mu_cols = img.mean(axis=0)                         # (W, C)
    return mu_global, mu_rows, mu_cols

def make_fitness(
    stats: TargetStats,
    spec: GenomeSpec,
    alpha: float,
    beta: float,
    gamma: float
) -> Callable[[list[float]], Tuple[float]]:
    """
    Devuelve una función DEAP-compatible: eval_fn(individual) -> (fitness, )
    Fitness combinada (nagativa de error cuadrático) con normalización por HC/WC ya incluída.
    """
    H, W, C = spec.H, spec.W, spec.C
    HC = H * C
    WC = W * C
    
    def eval_fn(genes: list[float]) -> Tuple[float]:
        # 1) genes -> imagen
        img = decode(genes, spec)  # (H, W, C) en [0, 1] si se usó normalización
        
        # 2) estadísticas de la imagen candidata
        mu_g , mu_r, mu_c = _candidate_stats(img)
        
        # 3) pérdidas
        loss_global = float(np.sum((mu_g - stats.mu_globabl) ** 2))  # escalar
        loss_rows = float(np.sum((mu_r - stats.mu_rows) ** 2)) / HC  # normalizado
        loss_cols = float(np.sum((mu_c - stats.mu_cols) ** 2)) / WC  # normalizado
        
        # 4) fitness = - combinación
        fitness = - (alpha * loss_global + beta * loss_rows + gamma * loss_cols)
        return (fitness,)
    
    return eval_fn