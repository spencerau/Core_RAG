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

        if not os.path.isabs(config_name):
            model_path = os.path.join(configs_dir, "model.yaml")
            if os.path.exists(model_path):
                model_config = _load_yaml(model_path)
                config = merge_configs(config, model_config)

        if os.environ.get('LOCAL_DEV', '').lower() == 'true':
            local_config_path = os.path.join(configs_dir, "config.local.yaml")
            if os.path.exists(local_config_path):
                config = merge_configs(config, _load_yaml(local_config_path))
            model_local_path = os.path.join(configs_dir, "model.local.yaml")
            if os.path.exists(model_local_path):
                config = merge_configs(config, _load_yaml(model_local_path))

        if 'QDRANT_HOST' in os.environ:
            config.setdefault('qdrant', {})['host'] = os.environ['QDRANT_HOST']

        if 'QDRANT_PORT' in os.environ:
            config.setdefault('qdrant', {})['port'] = int(os.environ['QDRANT_PORT'])

        if 'POSTGRES_PORT' in os.environ:
            config.setdefault('postgresql', {})['port'] = int(os.environ['POSTGRES_PORT'])

        _llm_host = os.environ.get('LLM_HOST') or os.environ.get('OLLAMA_HOST')
        if _llm_host:
            config.setdefault('embedding', {})['host'] = _llm_host
            if 'cluster' in config:
                config['cluster']['host'] = _llm_host

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
