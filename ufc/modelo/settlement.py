"""Taxonomia canonica y reglas minimas compartidas por targets y liquidacion.

La version forma parte del manifiesto del modelo. Las props concretas de una casa pueden
combinar categorias distinto; por eso ``other`` nunca se convierte silenciosamente en
decision.
"""

RULES_VERSION = "2026-08-02.1"

KO = "ko"
SUB = "sub"
DEC = "dec"
OTHER = "other"
INVALID = "invalid"


def canonical_method(method):
    """Metodo crudo UFCStats -> clase canonica, sin heuristica de fallback a decision."""
    value = str(method or "").strip().lower()
    if value in {"ko/tko", "tko - doctor's stoppage"}:
        return KO
    if value == "submission":
        return SUB
    if value.startswith("decision -"):
        return DEC
    if value in {"overturned"}:
        return INVALID
    if value in {"dq", "could not continue", "other"}:
        return OTHER
    return INVALID


def method_market_class(method, bookmaker="generic"):
    """Clase liquidable para el mercado KO/sub/decision; ``None`` exige regla externa."""
    canonical = canonical_method(method)
    # Regla versionada actual: doctor stoppage cuenta como KO/TKO. DQ/CNC/other depende
    # de la descripcion del mercado de la casa y no se adivina.
    if bookmaker in {"generic", "betano"} and canonical in {KO, SUB, DEC}:
        return canonical
    return None


def moneyline_result(outcome, selected_side):
    """Resultado generico de moneyline: win/loss/void para un lado ``a`` o ``b``."""
    if selected_side not in {"a", "b"}:
        raise ValueError("El lado debe ser 'a' o 'b'")
    raw = str(outcome or "").strip().upper()
    if raw not in {"W/L", "L/W"}:
        return "void"
    winner = "a" if raw == "W/L" else "b"
    return "win" if selected_side == winner else "loss"

