"""B6 — ST-GNN sin fuga frente a competidores clásicos (protocolo en preregistro.md).

Cada trabajo = (escala, modo, régimen, partición, pliegue). Para cada trabajo:
  * A / B = partición de hogares (aleatoria 50/50, o temporal en el régimen "tiempo");
  * A1 / A2 = partición interna de A para ajustar (entrada A1 -> objetivo A2; validación A2 -> A1);
  * se ajustan los 7 competidores con el mismo presupuesto (4 configuraciones) y se predice con
    entrada A; la verdad es B. Se guardan predicciones por zona y puertos mal ubicados en S0.

Uso:  python run_b6.py principal|kriging|tiempo [--K 10] [--modos auto,todos] [--escalas zat,utam] [--njobs 8]
No lanza NEOS: S0 se resuelve exacto por redondeo (separable por sede, como B5).
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
SAL = AQUI / "salidas"
CACHE = SAL / "cache_b6.pkl"
RES_H = [22, 23, 0, 1, 2, 3, 4, 5]
WORK_H = list(range(8, 18))
PUB_H = [18, 19, 20, 21]
CUOTAS = [7352, 2739, 779]
KNN = 8
METODOS = ["media_ciudad", "crudo_A", "EB", "RF_rezago", "IDW", "GCN", "STGNN"]
_D = None


def datos():
    global _D
    if _D is None:
        with open(CACHE, "rb") as f:
            _D = pickle.load(f)
    return _D


# ------------------------------------------------------------------ perfiles y vecindarios
def perfil(esc: str, modo: str, hh_mask: np.ndarray):
    """Conteos expandidos (N, 48: 24 salidas + 24 llegadas), reescalados a población, y n sin ponderar."""
    D = datos(); t = D["trips"]; Z = D[esc]; N = len(Z["cod"])
    sel = hh_mask[t["hh"]]
    if modo == "auto":
        sel = sel & t["auto"]
    esc_pob = D["hh"]["w"].sum() / max(D["hh"]["w"][hh_mask].sum(), 1e-9)
    C = np.zeros((N, 48)); n = np.zeros(N)
    ko, kd = ("u_ori", "u_des") if esc == "utam" else ("z_ori", "z_des")
    for kz, kh, off in ((ko, "h_sal", 0), (kd, "h_lle", 24)):
        m = sel & (t[kz] >= 0) & (t[kh] >= 0)
        np.add.at(C, (t[kz][m], t[kh][m] + off), t["w"][m] * esc_pob)
        np.add.at(n, t[kz][m], 1.0)
    return C, n


def omega_de(P):
    """Participación por ventana (residencial, trabajo, público) desde el perfil de 48; suma 1."""
    tot = P[:, :24] + P[:, 24:]
    W = np.c_[tot[:, RES_H].sum(1), tot[:, WORK_H].sum(1), tot[:, PUB_H].sum(1)]
    s = W.sum(1, keepdims=True)
    return np.where(s > 0, W / np.maximum(s, 1e-12), 1 / 3), s.ravel()


def covariables(esc: str, hh_mask: np.ndarray):
    D = datos(); h = D["hh"]; Z = D[esc]; N = len(Z["cod"])
    zi = h["utam"] if esc == "utam" else h["zat"]
    m = hh_mask & (zi >= 0)
    w = h["w"][m]; z = zi[m]
    def wmean(x):
        x = np.nan_to_num(x[m].astype(float), nan=np.nan)
        ok = np.isfinite(x)
        num = np.bincount(z[ok], weights=w[ok] * x[ok], minlength=N); den = np.bincount(z[ok], weights=w[ok], minlength=N)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    est = h["estrato"]
    X = np.c_[np.log1p(np.bincount(z, weights=w, minlength=N)), wmean(est), wmean((est <= 2).astype(float)),
              wmean((est >= 5).astype(float)), wmean(h["apto"]), wmean(h["pers"]), wmean(h["autos"]),
              Z["xy"][:, 0] / 1e4, Z["xy"][:, 1] / 1e4]
    sin_dato = ~np.isfinite(X).all(1)
    col_med = np.nanmean(X, 0)
    X = np.where(np.isfinite(X), X, col_med)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return np.c_[X, sin_dato.astype(float)]


class Obs:
    """Observación de entrada: perfil propio (oculto en `ocultas`), vecindario IDW entre zonas disponibles."""

    def __init__(self, esc, modo, hh_mask, ocultas=None):
        Z = datos()[esc]; self.N = len(Z["cod"])
        C, n = perfil(esc, modo, hh_mask)
        oc = np.zeros(self.N, bool) if ocultas is None else ocultas
        C[oc] = 0.0; n[oc] = 0.0
        self.C, self.n, self.oc = C, n, oc
        self.L = C.sum(1)
        self.P = np.where(self.L[:, None] > 0, C / np.maximum(self.L[:, None], 1e-12), 0.0)
        self.logL = np.log1p(self.L)
        self.avail = (n > 0)
        self.X = covariables(esc, hh_mask)
        xy = Z["xy"]; idx_av = np.where(self.avail)[0]
        from sklearn.neighbors import NearestNeighbors
        nn_ = NearestNeighbors(n_neighbors=min(KNN + 1, len(idx_av))).fit(xy[idx_av])
        d, j = nn_.kneighbors(xy)
        j = idx_av[j]
        es_self = j == np.arange(self.N)[:, None]
        d = np.where(es_self, np.inf, d)
        orden = np.argsort(d, 1)[:, :KNN]
        self.nb = np.take_along_axis(j, orden, 1); self.nd = np.maximum(np.take_along_axis(d, orden, 1), 50.0)

    def pesos(self, pot):
        w = self.nd ** (-pot)
        return w / w.sum(1, keepdims=True)

    def rezago(self, pot=1.0):
        w = self.pesos(pot) * self.n[self.nb]
        s = w.sum(1, keepdims=True)
        wn = np.where(s > 0, w / np.maximum(s, 1e-12), 1.0 / KNN)
        Plag = np.einsum("ik,ikf->if", wn, self.P[self.nb])
        llag = (wn * self.logL[self.nb]).sum(1)
        return Plag, llag, np.log1p(s.ravel())


# ------------------------------------------------------------------ métricas
def wmae_omega(P_hat, tgt: Obs, nodos):
    om_h, _ = omega_de(P_hat); om_t, s_t = omega_de(tgt.P)
    ok = nodos & (s_t > 0) & (tgt.L > 0)
    e = np.abs(om_h - om_t).mean(1)
    return float((tgt.L[ok] * e[ok]).sum() / tgt.L[ok].sum())


# ------------------------------------------------------------------ competidores clásicos
def pred_media(inp: Obs):
    Pc = inp.C.sum(0) / inp.C.sum()
    med = np.median(inp.logL[inp.avail])
    return np.tile(Pc, (inp.N, 1)), np.full(inp.N, med)


def pred_crudo(inp: Obs):
    Pm, lm = pred_media(inp)
    P = np.where(inp.avail[:, None], inp.P, Pm); l = np.where(inp.avail, inp.logL, lm)
    return P, l


def pred_eb(inp: Obs, k, kL, loc):
    nloc = loc.max() + 1
    Ploc = np.zeros((nloc, 48)); lloc = np.full(nloc, np.nan)
    for g in range(nloc):
        m = (loc == g) & inp.avail
        if m.any():
            c = inp.C[m].sum(0); Ploc[g] = c / c.sum()
            lloc[g] = np.average(inp.logL[m], weights=inp.n[m])
    Plag, llag, _ = inp.rezago(1.0)
    tiene = np.isfinite(lloc[loc])
    Ppr = np.where(tiene[:, None], Ploc[loc], Plag); lpr = np.where(tiene, lloc[loc], llag)
    n = inp.n[:, None]
    P = (n * inp.P + k * Ppr) / (n + k)
    l = (inp.n * inp.logL + kL * lpr) / (inp.n + kL)
    return P, l


def pred_idw(inp: Obs, lam, pot):
    w = inp.pesos(pot) * inp.n[inp.nb] * lam
    num = inp.n[:, None] * inp.P + np.einsum("ik,ikf->if", w, inp.P[inp.nb])
    den = inp.n + w.sum(1)
    Pm, lm = pred_media(inp)
    P = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-12), Pm)
    l = np.where(den > 0, (inp.n * inp.logL + (w * inp.logL[inp.nb]).sum(1)) / np.maximum(den, 1e-12), lm)
    return P, l


def rasgos_rf(inp: Obs, propias: bool):
    Plag, llag, ln = inp.rezago(1.0)
    partes = [inp.X, Plag, llag[:, None], ln[:, None]]
    if propias:
        partes += [inp.P, inp.logL[:, None], np.log1p(inp.n)[:, None]]
    return np.hstack(partes)


def ajustar_rf(pares, cfg, propias):
    from sklearn.ensemble import RandomForestRegressor
    Xs, Ps, ls, ws = [], [], [], []
    for inp, tgt, nodos in pares:
        m = nodos & (tgt.n > 0)
        Xs.append(rasgos_rf(inp, propias)[m]); Ps.append(tgt.P[m]); ls.append(tgt.logL[m]); ws.append(tgt.n[m])
    X, P, l, w = np.vstack(Xs), np.vstack(Ps), np.concatenate(ls), np.concatenate(ws)
    md, leaf = cfg
    rp = RandomForestRegressor(300, max_depth=md, min_samples_leaf=leaf, max_features=0.33, n_jobs=1, random_state=0).fit(X, P, sample_weight=w)
    rl = RandomForestRegressor(300, max_depth=md, min_samples_leaf=leaf, max_features=0.33, n_jobs=1, random_state=0).fit(X, l, sample_weight=w)
    def pred(inp):
        Xi = rasgos_rf(inp, propias)
        P = np.clip(rp.predict(Xi), 0, None); P = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
        return P, rl.predict(Xi)
    return pred


# ------------------------------------------------------------------ redes (GCN y ST-GNN)
def tensores(inp: Obs, torch, mu_l, sd_l):
    N = inp.N
    w = inp.pesos(1.0)
    filas = np.repeat(np.arange(N), KNN); cols = inp.nb.ravel()
    A = torch.sparse_coo_tensor(np.vstack([filas, cols]), w.ravel().astype(np.float32), (N, N)).coalesce()
    return {
        "A": A, "P": torch.tensor(inp.P * 24.0, dtype=torch.float32), "X": torch.tensor(inp.X, dtype=torch.float32),
        "l": torch.tensor((inp.logL - mu_l) / sd_l, dtype=torch.float32), "ln": torch.tensor(np.log1p(inp.n) / 5.0, dtype=torch.float32),
        "flag": torch.tensor((~inp.avail).astype(np.float32)),
    }


def construir_redes(torch):
    nn = torch.nn

    class Sage(nn.Module):
        def __init__(s, i, o):
            super().__init__(); s.a = nn.Linear(i, o); s.b = nn.Linear(i, o, bias=False)
        def forward(s, A, H):  # H: (N, ..., i)
            sh = H.shape
            agg = torch.sparse.mm(A, H.reshape(sh[0], -1)).reshape(sh)
            return s.a(H) + s.b(agg)

    class GCN(nn.Module):
        def __init__(s, ns, h):
            super().__init__(); s.g1 = Sage(ns + 48 + 3, h); s.g2 = Sage(h, h)
            s.op = nn.Linear(h, 48); s.ol = nn.Linear(h, 1)
        def forward(s, T, mask):
            keep = (~mask).float()[:, None]
            x = torch.cat([T["X"], T["P"] * keep, (T["l"] * keep.squeeze(1))[:, None], (T["ln"] * keep.squeeze(1))[:, None],
                           torch.maximum(T["flag"], mask.float())[:, None]], 1)
            h = torch.relu(s.g1(T["A"], x)); h = torch.relu(s.g2(T["A"], h))
            return s.op(h), s.ol(h).squeeze(1)

    class STGNN(nn.Module):
        def __init__(s, ns, h):
            super().__init__()
            s.g1 = Sage(5, h); s.g2 = Sage(h, h)
            s.st = nn.Sequential(nn.Linear(ns, h), nn.ReLU())
            s.gru = nn.GRU(2 * h, h, batch_first=True, bidirectional=True)
            s.oh = nn.Linear(2 * h, 2); s.ol = nn.Sequential(nn.Linear(2 * h + h, h), nn.ReLU(), nn.Linear(h, 1))
        def forward(s, T, mask):
            N = T["P"].shape[0]; keep = (~mask).float()
            P = T["P"] * keep[:, None]
            seq = torch.stack([P[:, :24], P[:, 24:]], 2)  # (N, 24, 2)
            fl = torch.maximum(T["flag"], mask.float())
            extra = torch.stack([fl, T["l"] * keep, T["ln"] * keep], 1)[:, None, :].expand(N, 24, 3)
            x = torch.cat([seq, extra], 2)
            h = torch.relu(s.g1(T["A"], x)); h = torch.relu(s.g2(T["A"], h))  # (N, 24, h)
            e = s.st(T["X"])
            y, _ = s.gru(torch.cat([h, e[:, None, :].expand(N, 24, e.shape[1])], 2))
            lo = s.oh(y)  # (N, 24, 2)
            logits = torch.cat([lo[:, :, 0], lo[:, :, 1]], 1)
            return logits, s.ol(torch.cat([y.mean(1), e], 1)).squeeze(1)

    return GCN, STGNN


def ajustar_red(tipo, pares_tr, par_val, cfg, semilla, epocas_fijas=None, kriging=False):
    import torch
    torch.set_num_threads(1); torch.manual_seed(semilla)
    GCN, STGNN = construir_redes(torch)
    allL = np.concatenate([p[1].logL[p[1].n > 0] for p in pares_tr])
    mu_l, sd_l = float(allL.mean()), float(allL.std() + 1e-9)
    def prep(par):
        inp, tgt, nodos = par
        T = tensores(inp, torch, mu_l, sd_l)
        m = nodos & (tgt.n > 0)
        return T, torch.tensor(tgt.P, dtype=torch.float32), torch.tensor((tgt.logL - mu_l) / sd_l, dtype=torch.float32), \
            torch.tensor(np.where(m, tgt.n, 0.0), dtype=torch.float32), torch.tensor(inp.oc)
    TR = [prep(p) for p in pares_tr]
    VA = prep(par_val) if par_val is not None else None
    h, wd = cfg
    ns = pares_tr[0][0].X.shape[1]
    model = (GCN if tipo == "GCN" else STGNN)(ns, h)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=wd)
    gen = np.random.default_rng(semilla)
    def perdida(out, P, l, w):
        logits, lh = out
        ce = -(P * torch.log_softmax(logits, 1)).sum(1)
        return ((ce + 0.5 * (lh - l) ** 2) * w).sum() / w.sum().clamp_min(1e-9)
    mejor, mejor_ep, estado, espera = np.inf, 0, None, 0
    max_ep = epocas_fijas or 400
    for ep in range(max_ep):
        model.train(); opt.zero_grad(); tot = 0.0
        for T, P, l, w, base_mask in TR:
            rnd = torch.tensor(gen.random(len(w)) < 0.25)
            out = model(T, base_mask | rnd)
            tot = tot + perdida(out, P, l, w)
        tot.backward(); opt.step()
        if VA is None:
            continue
        model.eval()
        with torch.no_grad():
            T, P, l, w, base_mask = VA
            vl = float(perdida(model(T, base_mask), P, l, w))
        if mejor - vl > 1e-5:
            mejor, mejor_ep, espera = vl, ep + 1, 0
            estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            espera += 1
            if espera >= 40:
                break
    if estado is not None:
        model.load_state_dict(estado)
    model.eval()
    def pred(inp):
        with torch.no_grad():
            T = tensores(inp, torch, mu_l, sd_l)
            logits, lh = model(T, torch.tensor(inp.oc))
            return torch.softmax(logits, 1).numpy().astype(float), lh.numpy() * sd_l + mu_l
    return pred, max(mejor_ep, 20)


# ------------------------------------------------------------------ S0 exacto
def s0_exacto(t):
    n = np.zeros_like(t)
    for v, q in enumerate(CUOTAS):
        b = np.floor(t[:, v]); r = int(round(q - b.sum())); nv = b.copy()
        if r > 0:
            nv[np.argsort(-(t[:, v] - b), kind="stable")[:r]] += 1
        n[:, v] = nv
    return n


def metas(om, L):
    prio = (L - L.min()) / (L.max() - L.min() + 1e-12); prio = np.maximum(prio, 1e-9)
    s = prio[:, None] * om
    return s / s.sum(0, keepdims=True) * np.array(CUOTAS)


# ------------------------------------------------------------------ trabajo
def particiones(regimen, k, H):
    D = datos(); rng = np.random.default_rng(k)
    if regimen == "tiempo":
        mes = D["hh"]["mes"]
        A = np.isin(mes, [3, 4, 5]); B = (mes == 6) if k == 0 else (mes == 10)
    else:
        A = rng.random(H) < 0.5; B = ~A
    r2 = np.random.default_rng(k + 100)
    A1 = A & (r2.random(H) < 0.5); A2 = A & ~A1
    return A, B, A1, A2


def trabajo(esc, modo, regimen, k, pliegue):
    t0 = time.time()
    D = datos(); H = D["H"]; Z = D[esc]; N = len(Z["cod"]); loc = Z["loc"]
    A, B, A1, A2 = particiones(regimen, k, H)
    kr = regimen == "kriging"
    if kr:
        g = np.random.default_rng(1000 + k).permutation(loc.max() + 1) % 5
        H_oc = g[loc] == pliegue
        pseudo = (~H_oc) & (np.random.default_rng(2000 + k).random(N) < 0.25)
    else:
        H_oc = np.zeros(N, bool); pseudo = np.zeros(N, bool)
    todos = np.ones(N, bool)
    # pares de ajuste: (entrada, objetivo, nodos de pérdida/evaluación)
    o_A1 = Obs(esc, modo, A1, H_oc); o_A2 = Obs(esc, modo, A2, H_oc)
    t_A1 = Obs(esc, modo, A1); t_A2 = Obs(esc, modo, A2)
    nodos_tr = ~H_oc
    pares_tr = [(o_A1, t_A2, nodos_tr), (o_A2, t_A1, nodos_tr)]
    if kr:
        v_in = Obs(esc, modo, A2, H_oc | pseudo); val = (v_in, t_A1, pseudo)
    else:
        val = (o_A2, t_A1, nodos_tr)
    tr_dir = [(o_A1, t_A2, nodos_tr)]
    inp = Obs(esc, modo, A, H_oc); tgt = Obs(esc, modo, B)
    eval_nodos = H_oc if kr else todos
    preds, elegido = {}, {}
    sc = lambda P: wmae_omega(P, val[1], val[2])
    preds["media_ciudad"] = pred_media(inp)
    preds["crudo_A"] = pred_crudo(inp)
    # EB: k y kL elegidos por separado (4 valores cada uno) en validación A2 -> A1
    rej = [5.0, 20.0, 50.0, 200.0]
    kb = min(rej, key=lambda kk: sc(pred_eb(val[0], kk, 20.0, loc)[0]))
    def err_l(l, par):
        tg, m = par[1], par[2] & (par[1].n > 0)
        return float((tg.L[m] * np.abs(l[m] - tg.logL[m])).sum() / tg.L[m].sum())
    kLb = min(rej, key=lambda kk: err_l(pred_eb(val[0], 20.0, kk, loc)[1], val))
    preds["EB"] = pred_eb(inp, kb, kLb, loc); elegido["EB"] = (kb, kLb)
    cfg_idw = [(0.5, 1.0), (2.0, 1.0), (0.5, 2.0), (2.0, 2.0)]
    ci = min(cfg_idw, key=lambda c: sc(pred_idw(val[0], *c)[0]))
    preds["IDW"] = pred_idw(inp, *ci); elegido["IDW"] = ci
    cfg_rf = [(None, 3), (None, 10), (12, 3), (12, 10)]
    propias = not kr
    best = min(cfg_rf, key=lambda c: sc(ajustar_rf(tr_dir, c, propias)(val[0])[0]))
    preds["RF_rezago"] = ajustar_rf(pares_tr, best, propias)(inp); elegido["RF_rezago"] = best
    cfg_nn = [(16, 1e-4), (32, 1e-4), (16, 1e-3), (32, 1e-3)]
    for tipo in ("GCN", "STGNN"):
        res = []
        for c in cfg_nn:
            f, ep = ajustar_red(tipo, tr_dir, val, c, semilla=k)
            res.append((sc(f(val[0])[0]), c, ep))
        _, c, ep = min(res, key=lambda r: r[0])
        f, _ = ajustar_red(tipo, pares_tr, None, c, semilla=k, epocas_fijas=ep)
        preds[tipo] = f(inp); elegido[tipo] = (c, ep)
    # salida por zona
    om_t, s_t = omega_de(tgt.P)
    filas = []
    for mth in METODOS:
        P, l = preds[mth]
        om, _ = omega_de(P)
        for i in np.where(eval_nodos)[0]:
            filas.append((esc, modo, regimen, k, pliegue, mth, i, int(loc[i]), tgt.L[i], tgt.n[i], s_t[i] > 0,
                          *om[i], *om_t[i], l[i], tgt.logL[i], inp.n[i]))
    df = pd.DataFrame(filas, columns=["escala", "modo", "regimen", "particion", "pliegue", "metodo", "zona", "loc", "L_B", "n_B", "ventana_B",
                                      "om_res", "om_tra", "om_pub", "omB_res", "omB_tra", "omB_pub", "logL_hat", "logL_B", "n_A"])
    # S0 a escala UTAM (solo régimen sin enmascarar: todas las zonas con predicción)
    s0 = []
    if not kr:
        Ub = Obs("utam", modo, B) if esc == "zat" else tgt
        om_b, _ = omega_de(Ub.P); nB = s0_exacto(metas(om_b, Ub.L))
        for mth in METODOS:
            P, l = preds[mth]
            Cn = P * np.expm1(np.maximum(l, 0))[:, None]
            if esc == "zat":
                Cu = np.zeros((len(datos()["utam"]["cod"]), 48)); np.add.at(Cu, Z["utam_idx"], Cn)
            else:
                Cu = Cn
            Lu = Cu.sum(1); Pu = Cu / np.maximum(Lu[:, None], 1e-12)
            om_u, _ = omega_de(Pu)
            nm = s0_exacto(metas(om_u, Lu))
            s0.append((esc, modo, regimen, k, mth, 0.5 * np.abs(nm - nB).sum()))
    s0 = pd.DataFrame(s0, columns=["escala", "modo", "regimen", "particion", "metodo", "puertos_mal_ubicados"])
    meta = {"escala": esc, "modo": modo, "regimen": regimen, "particion": k, "pliegue": pliegue, "seg": time.time() - t0,
            "elegido": {m: str(v) for m, v in elegido.items()}}
    return df, s0, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("regimen", choices=["principal", "kriging", "tiempo"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--modos", default="auto,todos")
    ap.add_argument("--escalas", default="zat,utam")
    ap.add_argument("--njobs", type=int, default=8)
    ap.add_argument("--etiqueta", default="")
    a = ap.parse_args()
    from joblib import Parallel, delayed
    tareas = []
    for esc in a.escalas.split(","):
        for modo in a.modos.split(","):
            if a.regimen == "tiempo":
                tareas += [(esc, modo, "tiempo", k, 0) for k in (0, 1)]  # 0: junio, 1: octubre
            elif a.regimen == "kriging":
                tareas += [(esc, modo, "kriging", k, f) for k in range(a.K) for f in range(5)]
            else:
                tareas += [(esc, modo, "principal", k, 0) for k in range(a.K)]
    tareas.sort(key=lambda t: t[0] != "zat")  # las ZAT (más lentas) primero
    t0 = time.time()
    print(f"{len(tareas)} trabajos, {a.njobs} procesos", flush=True)
    res = Parallel(n_jobs=a.njobs, verbose=5)(delayed(trabajo)(*t) for t in tareas)
    nombre = f"{a.regimen}{a.etiqueta}"
    pd.concat([r[0] for r in res]).to_csv(SAL / f"pred_{nombre}.csv.gz", index=False)
    s0 = [r[1] for r in res if len(r[1])]
    if s0:
        pd.concat(s0).to_csv(SAL / f"s0_{nombre}.csv", index=False)
    pd.DataFrame([r[2] for r in res]).to_csv(SAL / f"meta_{nombre}.csv", index=False)
    print(f"listo en {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
