"""
Patch openfold and cheap-proteins installed packages to make the missing
attn_core_inplace_cuda CUDA extension a soft error instead of a hard crash.
This extension is only needed for openfold's fused attention path, which
DiMA does not use.
"""
import pathlib

SITE = pathlib.Path('/opt/conda/envs/dima_env/lib/python3.10/site-packages')
OLD = 'attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")'
NEW = (
    'try:\n'
    '    attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")\n'
    'except ModuleNotFoundError:\n'
    '    attn_core_inplace_cuda = None'
)

targets = [
    SITE / 'openfold/utils/kernel/attention_core.py',
    SITE / 'openfold/model/structure_module.py',
    SITE / 'cheap/esmfold/_structure_module.py',
]

for p in targets:
    if not p.exists():
        print(f'MISSING: {p}')
        continue
    txt = p.read_text()
    if OLD in txt:
        p.write_text(txt.replace(OLD, NEW))
        print(f'Patched: {p.name}')
    else:
        print(f'Already patched or pattern missing: {p.name}')
