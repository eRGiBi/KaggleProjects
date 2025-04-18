import yaml

def log_exp(exp_id, params):

    with open('results/' + exp_id + '_result.yml', 'w') as yaml_file:
        yaml.dump(params, yaml_file, default_flow_style=True)
