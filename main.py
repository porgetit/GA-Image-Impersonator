from __future__ import annotations
from typing import Dict
import numpy as np
from deap import tools
import core
import image_task as task
from PIL import Image

def build_config() -> Dict:
    return {
        "image": {"path": "target.png", "rezise_to": [64, 64], "normalize": True},
        "genome": {"bounds": [0.0, 1.0]},
        "fitness": {"alpha": 0.1, "beta": 0.45, "gamma": 0.45},
        "ga": {
            "population_size": 80,
            "max_generations": 200,
            "cxpb": 0.9,
            "mutpb": 0.2,
            "hof_k": 1,
            "seed": 42,
            "cfg_ops": {
                "selection": {"type": "tournament", "k": 3},
                "crossover": {"type": "uniform", "indpb": 0.5},
                "mutation": {"type": "gaussian", "mu": 0.0, "sigma": 0.08, "indpb": 0.1},
            },
        },
        "output": {"save_best_image": True, "path": "best_image.png"},
    }
    
def main(config: Dict | None = None) -> None:
    cfg = config or build_config()
    
    # RNG consistente
    seed = cfg["ga"]["seed"]
    np.random.default_rng(seed)
    
    # 1) Cargar imagen objetivo
    I = task.load_image(
        path=cfg["image"]["path"],
        to_shape=tuple(cfg["image"]["rezise_to"]) if cfg["image"]["rezise_to"] else None,
        normalize=cfg["image"]["normalize"]
    )
    H, W, C = I.shape
    
    # 2) Stats objetivo
    stats = task.prepare_target_stats(I)
    
    # 3) Especificación del genoma (píxeles)
    low, high = cfg["genome"]["bounds"]
    spec = task.make_genome_spec(H=H, W=W, C=C, low=low, high=high)
    
    # 4) Fitness
    eval_fn = task.make_fitness(
        stats=stats,
        spec=spec,
        alpha=cfg["fitness"]["alpha"],
        beta=cfg["fitness"]["beta"],
        gamma=cfg["fitness"]["gamma"]
    )
    
    # 5) Toolbox
    toolbox = core.make_toolbox(
        gen_dim=spec.gene_dim,
        bounds=(low, high),
        eval_fn=eval_fn,
        cfg_ops=cfg["ga"]["cfg_ops"],
        seed=seed
    )
    
    # 6) Evolución
    hof, logbook, pop = core.evolve(
        toolbox=toolbox,
        population_size=cfg["ga"]["population_size"],
        max_generations=cfg["ga"]["max_generations"],
        cxpb=cfg["ga"]["cxpb"],
        mutpb=cfg["ga"]["mutpb"],
        hof_k=cfg["ga"]["hof_k"],
        verbose=True
    )
    
    best = hof[0]
    print(f"Mejor fitness: {best.fitness.values[0]}")
    
    # 7) Guardar mejor imagen
    if cfg["output"]["save_best_image"]:
        img_best = task.decode(best, spec) # best es un individuo (lista de genes); decode acepta list[float]
        # Des-normalizar a [0, 255]
        img_u8 = np.clip(img_best * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img_u8).save(cfg["output"]["path"])
        print(f"Mejor imagen guardada en: {cfg['output']['path']}")
        
        
if __name__ == "__main__":
    main()