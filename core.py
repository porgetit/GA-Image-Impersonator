from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from deap import base, creator, tools, algorithms

# Tipos
Genes = list[float]
EvalFn = Callable[[Genes], Tuple[float]]

def _ensure_creators():
    """
    Crea tipo de deap si no existen aún.
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
def make_toolbox(
    gen_dim: int,
    bounds: Tuple[float, float],
    eval_fn: EvalFn,
    cfg_ops: Dict,
    seed: Optional[int] = None,
) -> base.Toolbox:
    """
    Construye y devuelve un toolbox de DEAP:
    - attibute: generador uniforme en [low, high]
    - individual: lista de longitud gen_dim
    - population: lista de individuos
    - evaluate: eval_fn inyectada
    - mate/mutate/select: según cfg_ops
    """
    
    _ensure_creators()
    
    low, high = bounds
    rng = np.random.default_rng(seed)
    
    toolbox = base.Toolbox()
    
    # Atributo base: flotante uniforme en [low, high]
    toolbox.register("attr_float", lambda: float(rng.uniform(low, high)))
    
    # Individuo y población
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=gen_dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluación
    toolbox.register("evaluate", eval_fn)
    
    # Selección
    sel_type = cfg_ops.get("selection", {}).get("type", "tournament")
    if sel_type == "tournament":
        k = cfg_ops["selection"].get("k", 3)
        toolbox.register("select", tools.selTournament, tournsize=k)
    elif sel_type == "roulette":
        toolbox.register("select", tools.selRoulette)
    elif sel_type == "rank":
        toolbox.register("select", tools.selBest) 
    else:
        raise NotImplementedError(f"Función de selección '{sel_type}' no implementada.")
    
    # Cruce
    cx_cfg = cfg_ops.get("crossover", {"type": "uniform", "indpb": 0.5})
    cx_type = cx_cfg.get("type", "uniform")
    if cx_type == "uniform":
        indpb = cx_cfg.get("indpb", 0.5)
        toolbox.register("mate", tools.cxUniform, indpb=indpb)
    elif cx_type == "one_point":
        toolbox.register("mate", tools.cxOnePoint)
    elif cx_type == "two_point":
        toolbox.register("mate", tools.cxTwoPoint)
    else:
        raise NotImplementedError(f"Función de cruce '{cx_type}' no implementada.")
    
    # Mutación
    mut_cfg = cfg_ops.get("mutation", {"type": "gaussian", "mu": 0, "sigma": 0.05, "indpb": 0.1})
    mut_type = mut_cfg.get("type", "gaussian")
    if mut_type == "gaussian":
        mu = mut_cfg.get("mu", 0)
        sigma = mut_cfg.get("sigma", 0.05)
        indpb = mut_cfg.get("indpb", 0.1)
        
        def _mut_gaussian_clip(individual):
            tools.mutGaussian(individual, mu, sigma, indpb)
            # Clipping in place a [low, high]
            for i in range(len(individual)):
                if individual[i] < low: individual[i] = low
                elif individual[i] > high: individual[i] = high
            return individual,
        toolbox.register("mutate", _mut_gaussian_clip)
    
    elif mut_type == "uniform_reset":
        indpb = mut_cfg.get("indpb", 0.1)
        
        def _mut_uniform_reset_clip(individual):
            for i in range(len(individual)):
                if rng.random() < indpb:
                    individual[i] = float(rng.uniform(low, high))
            return individual,
        toolbox.register("mutate", _mut_uniform_reset_clip)

    else:
        raise NotImplementedError(f"Función de mutación '{mut_type}' no implementada.")
    
    return toolbox

def evolve(
    toolbox: base.Toolbox,
    population_size: int,
    max_generations: int,
    cxpb: float,
    mutpb: float,
    hof_k: int = 1,
    verbose: bool = True,
):
    """
    Ejecuta un ciclo evolutivo usando "eaSimple" de DEAP.
    Retorna (hof, logbook, population).
    """
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(hof_k, similar=np.allclose)
    
    # Estadísticas opcionales (promedio, std, min, max)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # TODO (opcional): personalizar el bucle en lugar de eaSimple si se requiere más control
    pop, logbook = algorithms.eaSimple(
        popuplation=pop,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=max_generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )
    
    return hof, logbook, pop