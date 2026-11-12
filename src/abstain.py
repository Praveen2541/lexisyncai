


def validate_claims(claims, ctx_ids):
    """True only if there is >=1 claim and every claim cites >=1 ID that exists in the context."""
    return bool(claims) and all(
        c.get("cites") and all(x in ctx_ids for x in c["cites"]) for c in claims)


def confidence(answerable, top, claims, ctx_ids, tau_low, tau_high):
    if not answerable or top < tau_low or not claims:
        return "INSUFFICIENT EVIDENCE"
    if not validate_claims(claims, ctx_ids):
        return "LOW"
    return "HIGH" if top >= tau_high else "MEDIUM"
