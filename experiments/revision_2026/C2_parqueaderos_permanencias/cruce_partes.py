#!/usr/bin/env python
"""Cruce Parte 1 × Parte 2: llegadas de conductores de auto en zonas sin candidatos, por cuartil de ν.

Requiere las salidas de parte1_parqueaderos.py, parte1b_vulnerabilidad.py y parte2_llegadas_permanencias.py.
"""
from pathlib import Path
import json
import pandas as pd

S = Path(__file__).resolve().parent / "salidas"
z = pd.read_csv(S / "llegadas_permanencia_por_zat.csv").merge(pd.read_csv(S / "cobertura_por_zat.csv"), on="ZAT")
tot = z.llegadas_expandidas.sum()
d = {
    "zat_sin_candidato": int((z.n_total == 0).sum()),
    "zat_sin_candidato_con_llegadas_muestra": int(((z.n_total == 0) & (z.llegadas_muestra > 0)).sum()),
    "pct_llegadas_expandidas_en_zat_sin_candidato": round(100 * z.loc[z.n_total == 0, "llegadas_expandidas"].sum() / tot, 1),
    "pct_llegadas_expandidas_en_zat_sin_capacidad_conocida": round(100 * z.loc[z.cupos_conocidos == 0, "llegadas_expandidas"].sum() / tot, 1),
}
u = pd.read_csv(S / "llegadas_permanencia_por_utam.csv").merge(pd.read_csv(S / "candidatos_por_utam_vulnerabilidad.csv"), on="UTAM")
q = u.groupby("cuartil_vulnerabilidad").agg(llegadas_expandidas=("llegadas_expandidas", "sum"),
                                           llegadas_muestra=("llegadas_muestra", "sum"),
                                           candidatos=("n_total", "sum"), cupos_conocidos=("cupos_conocidos", "sum"))
q["candidatos_por_1000_llegadas"] = (1000 * q.candidatos / q.llegadas_expandidas).round(2)
q["cupos_por_1000_llegadas"] = (1000 * q.cupos_conocidos / q.llegadas_expandidas).round(2)
q.reset_index().to_csv(S / "cruce_llegadas_candidatos_por_cuartil.csv", index=False)
d["por_cuartil"] = q.round(1).reset_index().to_dict(orient="records")
(S / "diagnostico_cruce.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
print(json.dumps(d, indent=2, ensure_ascii=False))
