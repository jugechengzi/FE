from .compute_z import compute_z as compute_z_base
from .compute_z import find_fact_lookup_idx
from .compute_z_mlp import compute_z as compute_z_mlp

# Z方法配置
Z_METHODS_CONFIG = {
    "all": {
        "compute_fn": compute_z_base,
        "strategy": "all_layers",
        "description": "Compute z at all specified edit layers independently"
    },
    "firstforward": {
        "compute_fn": compute_z_base,
        "strategy": "first_forward",
        "description": "Compute z at first layer, then propagate forward to others"
    },
    "mlp_all": {
        "compute_fn": compute_z_mlp,
        "strategy": "mlp_all_layers",
        "description": "Compute z at all specified edit layers independently using MLP"
    },
    "mlp_firstforward": {
        "compute_fn": compute_z_mlp,
        "strategy": "mlp_first_forward",
        "description": "Compute z at first layer using MLP, then propagate forward to others"
    }
}
    # 新增方法（用户后续添加）：
    # "all": {
    #     "compute_fn": compute_z_memit,
    #     "strategy": "all_layers",  # 计算所有编辑层的z
    #     "description": "Compute z at all edit layers"
    # },
    # "firstforward": {
    #     "compute_fn": compute_z_memit,
    #     "strategy": "first_forward",  # 计算第一层，然后前向传播得到其他层
    #     "description": "Compute z at first layer, then propagate forward"
    # }



def get_z_method_config(z_method: str) -> dict:
    """Get z method configuration."""
    if z_method not in Z_METHODS_CONFIG:
        raise ValueError(
            f"Unknown z_method: {z_method}. Available methods: {list(Z_METHODS_CONFIG.keys())}"
        )
    return Z_METHODS_CONFIG[z_method]


def get_z_compute_function(z_method: str):
    """Get z computation function by method name."""
    config = get_z_method_config(z_method)
    return config["compute_fn"]


def get_z_strategy(z_method: str) -> str:
    """Get computation strategy for z method."""
    config = get_z_method_config(z_method)
    return config["strategy"]


def list_z_methods():
    """List all available z methods."""
    methods = []
    for method, config in Z_METHODS_CONFIG.items():
        methods.append({
            "name": method,
            "strategy": config["strategy"],
            "description": config["description"]
        })
    return methods
