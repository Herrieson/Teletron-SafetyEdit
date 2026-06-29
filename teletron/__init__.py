try:
    import teletron.megatron_adaptor
except ModuleNotFoundError as exc:
    if exc.name != "megatron":
        raise
