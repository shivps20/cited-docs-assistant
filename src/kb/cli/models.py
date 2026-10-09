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


CHECK_TEXT = [{"role": "system", "content": "You answer in one word."},
              {"role": "user", "content": "Reply with the single word: OK"}]
CHECK_JSON = [{"role": "user", "content": 'Return a JSON object with one key "ok" set to true.'}]


def models_check(names: list[str] | None) -> int:
    """`kb models check`: send every configured model (or the named ones) one tiny text prompt and one
    JSON prompt; report reachability, speed, token counts and whether JSON came back.

    Models whose key is not set are skipped. The prompts are a few tokens each (cost: fractions of a cent).
    """
    import json

    from kb.llm.providers import LLMError, make_provider

    try:
        catalogue = load_catalogue()
        profiles = [catalogue.profile(n) for n in names] if names else list(catalogue.models.values())
    except CatalogueError as e:
        print(e)
        return 1

    print(f"{'name':<16} {'status':<8} {'seconds':>7} {'in':>6} {'out':>5} {'json':<5}  reply / problem")
    failed = 0
    for p in profiles:
        if not p.ready:
            print(f"{p.name:<16} {'skipped':<8} {'':>7} {'':>6} {'':>5} {'':<5}  {p.api_key_env} not set")
            continue
        provider = make_provider(p)
        try:
            text = provider.generate(CHECK_TEXT)
            reply = provider.generate(CHECK_JSON, json_format=True)
        except LLMError as e:
            failed += 1
            print(f"{p.name:<16} {'FAILED':<8} {'':>7} {'':>6} {'':>5} {'':<5}  {e}")
            continue
        try:
            json_ok = json.loads(reply.text).get("ok") is True
        except (ValueError, AttributeError):
            json_ok = False
        print(f"{p.name:<16} {'ok':<8} {text.seconds:>7.1f} {text.prompt_tokens or '-':>6} {text.output_tokens or '-':>5} "
              f"{'yes' if json_ok else 'NO':<5}  {text.text.strip()[:40]!r} ({text.model})")
        if not json_ok:
            print(f"WARN {p.name}: the JSON reply was not a JSON object with ok=true: {reply.text[:80]!r}; "
                  "set json_mode: false for this model if its endpoint ignores response_format")
    print(f"\n{len(profiles)} model(s) checked, {failed} failed")
    return 1 if failed else 0
