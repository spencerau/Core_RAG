import os
import yaml


def get_project_root():
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if os.path.exists(os.path.join(current_dir, "configs")):
        return current_dir
    
    if os.path.exists("/app/configs"):
        return "/app"
    
    return os.getcwd()

def load_config(config_name=None):
    if config_name is None:
        config_name = os.environ.get('CONFIG_FILE', 'config.yaml')
    
    project_root = get_project_root()
    
    if os.path.isabs(config_name):
        config_path = config_name
    else:
        config_path = os.path.join(project_root, "configs", config_name)
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        print(f"Loaded config: {config_path}")
        
        local_config_path = os.path.join(project_root, "configs", "config.local.yaml")
        if os.path.exists(local_config_path):
            with open(local_config_path, 'r') as file:
                local_config = yaml.safe_load(file)
                config = merge_configs(config, local_config)
                print(f"Loaded local config overrides from {local_config_path}")
        
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