"""Model locations and explicitly reviewed upgrades of frozen service code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_COMMIT = '4409abe925c904e570be78ebfb5e77ac3491dff8'

# Only these exact revisions preserve the legacy MLP input and settlement path.
COMPATIBLE_REVISIONS = {
    ('etf_ppo', 'c7bc071e1cd898bbaf380534769f84789d32078fbf336f9bd8828240aa065bf3'): '0330fd78a416bef9dc8e100c30a9d25ed776af4669485395992c15ba8763faa4',
    ('daily', 'ff9f0b907342fb1aa43eeba8f7b2a74630f3c48e324e5d12a58a0f0dda1478b6'): '2b4c630c6aa41b643bfd855f3d62433aca22ba686ffbc5d08079a67a24f4855d',
    ('daily', '5fd119a215a3cff93802a2e5e86b4c26e81eeaa50ba01d38dacec8a2fa64770d'): '2b4c630c6aa41b643bfd855f3d62433aca22ba686ffbc5d08079a67a24f4855d',
    ('simulation', '65231d1532fded6df3828effb25bd3e01cde98ba7071c940f7f6fa92396bde5c'): '17ba6ff94fcb91dc6073c0412d12b625c22500e9889c2596bf3e2f183e0ed8fb',
    ('simulation', '66be1faeb337730d6b919d181ab87ee58328be54de0294a1d0d562e9d77bb86b'): '17ba6ff94fcb91dc6073c0412d12b625c22500e9889c2596bf3e2f183e0ed8fb',
}


def revision_matches(component, expected, actual):
    return actual == expected or COMPATIBLE_REVISIONS.get((component, expected)) == actual
