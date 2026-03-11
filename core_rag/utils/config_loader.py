import os
import yaml


def get_project_root():
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if os.path.exists(os.path.join(current_dir, "configs")):
        return current_dir

    if os.path.exists("/app/configs"):
        return "/app"

    return os.getcwd()

def _load_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file) or {}

def load_config(config_name=None):
    if config_name is None:
        config_name = os.environ.get('CONFIG_FILE', 'config.yaml')

    project_root = get_project_root()
    configs_dir = os.path.join(project_root, "configs")

    if os.path.isabs(config_name):
        config_path = config_name
    else:
        config_path = os.path.join(configs_dir, config_name)

    try:
        config = _load_yaml(config_path)

        # Merge model.yaml if it exists alongside config.yaml
        if not os.path.isabs(config_name):
            model_path = os.path.join(configs_dir, "model.yaml")
            if os.path.exists(model_path):
                model_config = _load_yaml(model_path)
                config = merge_configs(config, model_config)

        # Apply local overrides (can override both config.yaml and model.yaml settings)
        local_config_path = os.path.join(configs_dir, "config.local.yaml")
        if os.path.exists(local_config_path):
            local_config = _load_yaml(local_config_path)
            config = merge_configs(config, local_config)

        if 'QDRANT_HOST' in os.environ:
            config['qdrant']['host'] = os.environ['QDRANT_HOST']

        if 'OLLAMA_HOST' in os.environ:
            if 'embedding' not in config:
                config['embedding'] = {}
            config['embedding']['ollama_host'] = os.environ['OLLAMA_HOST']

            if 'cluster' in config:
                config['cluster']['ollama_host'] = os.environ['OLLAMA_HOST']

        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error parsing YAML file: {e}")


def merge_configs(base_config, override_config):
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config
