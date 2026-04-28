def safe_get(data, path, default=None):
    try:
        keys = path.split(".")
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, None)
            else:
                return default
        return data if data is not None else default
    except:
        return default