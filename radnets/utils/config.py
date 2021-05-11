import yaml

def load_config(filename):
    """
    Load a `radnets` YAML config file.

    Parameters
    ----------
    filename : str
        Filename of `radnets` YAML config file.

    Returns
    -------
    config : dict
        Dinctionary from YAML file.
    """
    with open(filename, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    return config

def get_filename(config, path):
    """
    Get the filename corresponding to a given `path` in the dictionary `config`.

    Parameters
    ----------
    config : dict
        Dictionary to obtain filename from.
    path : str
        String in the form 'key1/key2/.../keyn' required to reach filename.
        At the leaf node, there should be a 'path' and 'file' key.

    Example `path='detection/results'` would lead to the `results` key inside
    of the dictionary under the the `detection` dictionary inside of `config`:
    detection:
      results:
        path: /path/to/file
        file: filename

    Returns
    -------
    filename : str
    """
    keys = path.split('/')
    node = config
    for key in keys:
        node = node[key]

    file_path = node['path']
    file_name = node['file']
    return f'{file_path}/{file_name}'
