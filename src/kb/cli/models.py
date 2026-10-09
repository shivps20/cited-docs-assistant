"""`kb models`: the model catalogue (config/models.yaml)."""

from kb.llm.catalogue import ROLES, CatalogueError, load_catalogue


def models_list() -> int:
    """`kb models list`: validate the catalogue and show every model, its role(s) and whether its key is set."""
    try:
        catalogue = load_catalogue()
    except CatalogueError as e:
        print(e)
        return 1

    roles_of = {name: [r for r in ROLES if catalogue.roles[r] == name] for name in catalogue.models}
    if catalogue.source == "settings":
        print("no models.yaml: catalogue built from LLM_MODEL / OPENAI_* / KB_LLM_PROVIDER "
              "(copy config/models.example.yaml to config/models.yaml to configure)\n")
    else:
        print(f"catalogue: {catalogue.source}\n")
    print(f"{'name':<16} {'adapter':<18} {'location':<9} {'key':<8} {'context':>8} {'output':>7}  model / roles")
    for p in catalogue.models.values():
        key = "-" if not p.api_key_env else ("set" if p.ready else "NO KEY")
        roles = ", ".join(roles_of[p.name] + (["fallback"] if p.name == catalogue.fallback else []))
        print(f"{p.name:<16} {p.adapter:<18} {p.location:<9} {key:<8} {p.context_tokens:>8} {p.max_output_tokens:>7}  "
              f"{p.model}{f'  [{roles}]' if roles else ''}")
    for warning in catalogue.warnings:
        print(f"WARN {warning}")
    missing = [p.name for p in catalogue.models.values() if not p.ready]
    for name in missing:
        print(f"WARN {name}: {catalogue.models[name].api_key_env} is not set in the environment or .env")
    print(f"\nok: {len(catalogue.models)} models, {len(missing)} without a key")
    return 0
