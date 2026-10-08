"""X5 RetailHero, версия 3.
1) ~140 признаков, как у участников соревнования (чеки, время, бонусы, магазины, категории товаров).
2) Ваш метод получает те же признаки, что и ML-модели:
   - «сегменты»: эффект по сегментам каждой фичи, фичи с реальной разницей эффекта весят больше;
   - «дерево»: делим там, где разница эффекта между группами больше всего (с учётом шума), до 8 сегментов;
   - «лес»: много таких деревьев, усреднение — для максимального качества ранжирования.
   Везде эффект считается через мешочки: клиент сравнивается со своей историей, а main — с control.
3) Сравнение с настроенными ML и Causal Forest, метрика соревнования uplift@30%.
4) Отчёт для маркетолога: сегменты дерева, эффект в п.п., интервалы, деньги.
Нужны в DATA: purchases.csv, clients.csv, products.csv, uplift_train.csv.
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as R, HistGradientBoostingClassifier as C
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings('ignore')

DATA = '/content/x5'
WINDOW_DAYS = 7
SEEDS = 5                    # повторов разбиения train/test
USE_CAUSAL_FOREST = True     # !pip install econml
MAX_LEAVES = 8               # сколько сегментов максимум в дереве для маркетолога
MIN_ARM = 1500               # минимум клиентов в main и в control в сегменте дерева
N_TREES = 60                 # деревьев в лесу
SMS_COST, PROFIT_PER_BUYER = 2.0, 150.0   # <-- подставьте свои цены
N_MC = 300
# Признаки, по которым дерево для маркетолога может делить клиентов. None = все ~140.
# Пример понятного маркетологу набора (раскомментируйте, если дерево делит по «странным» признакам):
# TREE_FEATS = ['active_weeks_share', 'n_trans', 'mean_check', 'days_since_last', 'issue_days', 'redeemed',
#               'age', 'gender_F', 'gender_M', 'share_weekend', 'share_evening', 'alco_share', 'own_share',
#               'trend_4w', 'n_stores', 'regular_points_spent']
TREE_FEATS = None

def path(name):
    for ext in ['', '.gz']:
        p = os.path.join(DATA, name + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f'{name} не найден в {DATA}')

# =====================================================================================
# 1. Признаки
# =====================================================================================
t0 = time.time()
train = pd.read_csv(path('uplift_train.csv'))
clients = pd.read_csv(path('clients.csv')).set_index('client_id').loc[train.client_id]
products = pd.read_csv(path('products.csv')).set_index('product_id')
ids = set(train.client_id)
L2 = sorted(products.level_2.dropna().unique()); L1 = sorted(products.level_1.dropna().unique())
TX_COLS = ['client_id', 'transaction_id', 'transaction_datetime', 'purchase_sum', 'store_id',
           'regular_points_received', 'express_points_received', 'regular_points_spent', 'express_points_spent']
tx_parts, prod_parts = [], []
for ch in pd.read_csv(path('purchases.csv'), chunksize=3_000_000):
    ch = ch[ch.client_id.isin(ids)]
    tx_parts.append(ch[TX_COLS].drop_duplicates('transaction_id'))
    pr = products.reindex(ch.product_id)
    spend = ch.trn_sum_from_iss.fillna(0).values
    agg = pd.DataFrame({'client_id': ch.client_id.values, 'rows': 1, 'qty': ch.product_quantity.fillna(0).values,
                        'spend': spend, 'alco': spend * pr.is_alcohol.fillna(0).values,
                        'own': spend * pr.is_own_trademark.fillna(0).values,
                        'netto': pr.netto.fillna(0).values, 'red_sum': ch.trn_sum_from_red.fillna(0).values})
    for i, l in enumerate(L1): agg[f'l1_{i}'] = spend * (pr.level_1.values == l)
    for i, l in enumerate(L2): agg[f'l2_{i}'] = spend * (pr.level_2.values == l)
    prod_parts.append(agg.groupby('client_id').sum())
    print(f'  прочитано {sum(len(p) for p in tx_parts):,} чеков…', flush=True)
tx = pd.concat(tx_parts).drop_duplicates('transaction_id')
pa = pd.concat(prod_parts).groupby(level=0).sum()
tx['dt'] = pd.to_datetime(tx.transaction_datetime)
print(f'Покупки: {len(tx):,} чеков, {time.time() - t0:.0f} c')

end = tx.dt.max().normalize() + pd.Timedelta(days=1)
K = int((end - tx.dt.min().normalize()).days // WINDOW_DAYS); first = end - pd.Timedelta(days=K * WINDOW_DAYS)
tx = tx[tx.dt >= first]; tx['w'] = ((tx.dt - first).dt.days // WINDOW_DAYS).astype(int)
cidx = pd.Series(np.arange(len(train)), index=train.client_id); ci = tx.client_id.map(cidx).values
bought = np.zeros((len(train), K), np.float32); bought[ci, tx.w.values] = 1
cnt_w = np.zeros((len(train), K), np.float32); np.add.at(cnt_w, (ci, tx.w.values), 1)
issue = pd.to_datetime(clients.first_issue_date)
arrive = np.ceil(np.clip(((issue.values - np.datetime64(first)) / np.timedelta64(WINDOW_DAYS, 'D')).astype(float), 0, None)).astype(int)
H = np.where(np.arange(K)[None, :] >= arrive[:, None], bought, np.nan)   # мешочки
L = (~np.isnan(H)).sum(1)

g = tx.groupby('client_id')
ps = g.purchase_sum
f = pd.DataFrame({'n_trans': g.size(), 'sum': ps.sum(), 'mean_check': ps.mean(), 'std_check': ps.std(),
                  'max_check': ps.max(), 'min_check': ps.min(), 'n_stores': g.store_id.nunique(),
                  'days_since_first': (end - g.dt.min()).dt.days, 'days_since_last': (end - g.dt.max()).dt.days})
for c in ['regular_points_received', 'express_points_received', 'regular_points_spent', 'express_points_spent']:
    f[c] = g[c].sum(); f[c + '_share_trans'] = g[c].apply(lambda s: (s != 0).mean())
top_store = tx.groupby(['client_id', 'store_id']).size().groupby(level=0).max()
f['top_store_share'] = top_store / f.n_trans
h = tx.dt.dt.hour; wd = tx.dt.dt.weekday
for lab, m in [('night', h < 7), ('morning', (h >= 7) & (h < 12)), ('day', (h >= 12) & (h < 18)), ('evening', h >= 18)]:
    f[f'share_{lab}'] = m.groupby(tx.client_id).mean()
for d in range(7): f[f'share_wd{d}'] = (wd == d).groupby(tx.client_id).mean()
f['share_weekend'] = (wd >= 5).groupby(tx.client_id).mean()
gaps = tx.sort_values('dt').groupby('client_id').dt.apply(lambda s: s.diff().dt.days.mean())
f['mean_gap_days'] = gaps
for wks in [1, 2, 4]:
    m = tx.dt >= end - pd.Timedelta(days=7 * wks)
    f[f'trans_last_{wks}w'] = m.groupby(tx.client_id).sum(); f[f'sum_last_{wks}w'] = tx.purchase_sum.where(m, 0).groupby(tx.client_id).sum()
prev = (tx.dt < end - pd.Timedelta(days=28)) & (tx.dt >= end - pd.Timedelta(days=56))
f['sum_prev_4w'] = tx.purchase_sum.where(prev, 0).groupby(tx.client_id).sum()
f['trend_4w'] = (f.sum_last_4w + 1) / (f.sum_prev_4w + 1)
red = pd.to_datetime(clients.first_redeem_date).reindex(tx.client_id).values
after = tx.dt.values >= red
f['trans_after_redeem'] = pd.Series(after, index=tx.index).groupby(tx.client_id).sum()
f = f.reindex(train.client_id)
pa = pa.reindex(train.client_id)
for c in ['alco', 'own', 'red_sum'] + [c for c in pa if c.startswith('l1_') or c.startswith('l2_')]:
    f[c + '_share'] = (pa[c] / pa.spend.replace(0, np.nan)).values
f['items_per_trans'] = (pa.rows.values / f.n_trans.values); f['qty_per_trans'] = (pa.qty.values / f.n_trans.values)
f['avg_netto'] = (pa.netto / pa.rows.replace(0, np.nan)).values
f['age'] = clients.age.where(clients.age.between(14, 100)).values
for gcode in ['F', 'M', 'U']: f[f'gender_{gcode}'] = (clients.gender.values == gcode).astype(int)
f['issue_days'] = (end - issue).dt.days.values
f['redeem_days'] = (end - pd.to_datetime(clients.first_redeem_date)).dt.days.values
f['issue_to_redeem'] = f.issue_days - f.redeem_days; f['redeemed'] = clients.first_redeem_date.notna().astype(int).values
for k in range(K): f[f'week_{k}'] = cnt_w[:, k]
f['active_weeks_share'] = np.nanmean(H, 1); f['n_windows'] = L
f = f.reset_index(drop=True).astype(float)
t = train.treatment_flg.values; y = train.target.values.astype(float)
keep = L >= 2
f, H, L, t, y = f[keep].reset_index(drop=True), H[keep], L[keep], t[keep], y[keep]
N, KW = len(y), H.shape[1]
FEATS = list(f.columns); X = f.values
FT = [FEATS.index(c) for c in TREE_FEATS] if TREE_FEATS else None
# исход, скорректированный мешочком: после − ожидаемое «обычное» (точка, к которой сходится Монте-Карло)
Z = y - np.nanmean(H, 1)
print(f'Признаков: {len(FEATS)}, клиентов: {N:,}, окон: {K}; main {y[t == 1].mean():.3f} vs control {y[t == 0].mean():.3f}; '
      f'{time.time() - t0:.0f} c')

# =====================================================================================
# 2. Монте-Карло (для интервалов) и метрики
# =====================================================================================
def mc_effect(rows, rng, n_mc=N_MC):
    """Эффект в п.п. (разница): бутстрэп клиентов + вытаскивание из мешочка (U → элемент), main − control."""
    m, c = rows[t[rows] == 1], rows[t[rows] == 0]
    if min(len(m), len(c)) < 30: return None
    res = []
    for ids_ in (m, c):
        Hs, Ls = H[ids_], L[ids_]
        b = rng.integers(0, len(ids_), (n_mc, len(ids_)), dtype=np.int32)
        d = Hs[b, KW - Ls[b] + (rng.random(b.shape) * Ls[b]).astype(int)]
        res.append((y[ids_][b] - d).mean(1))
    return res[0] - res[1]

def qini(yv, tv, s):
    o = np.argsort(-s, kind='stable'); yv, tv = yv[o], tv[o]
    nt, nc = np.cumsum(tv), np.cumsum(1 - tv)
    q = np.cumsum(yv * tv) - np.cumsum(yv * (1 - tv)) * np.divide(nt, nc, out=np.zeros(len(nt)), where=nc > 0)
    return (q - np.linspace(0, q[-1], len(q))).mean() / len(q) * 1000
def uplift_at30(yv, tv, s):
    top = np.argsort(-s, kind='stable')[: int(len(s) * .3)]
    return yv[top][tv[top] == 1].mean() - yv[top][tv[top] == 0].mean()

# =====================================================================================
# 3. Ваш метод на богатых признаках
# =====================================================================================
def effect_stats(z, tt):
    """Эффект (main − control по скорректированному исходу) и его дисперсия."""
    a, b = z[tt == 1], z[tt == 0]
    if min(len(a), len(b)) < 2: return np.nan, np.inf
    return a.mean() - b.mean(), a.var() / len(a) + b.var() / len(b)

def thresholds(x, k=9):
    q = np.unique(np.nanquantile(x, np.linspace(.1, .9, k)))
    return q[:-1] if len(q) > 1 and q[-1] == np.nanmax(x) else q

# --- 3a. Сегменты каждой фичи, взвешенные по реальности различий ---
def segments_score(tr, te):
    zt, tt = Z[tr], t[tr]; e0, _ = effect_stats(zt, tt)
    num, den = np.zeros(len(te)), 0.0
    for j in range(X.shape[1]):
        cuts = thresholds(X[tr, j], 3)
        btr = np.where(np.isnan(X[tr, j]), -1, np.searchsorted(cuts, X[tr, j], side='right'))
        bte = np.where(np.isnan(X[te, j]), -1, np.searchsorted(cuts, X[te, j], side='right'))
        eff, Q, dfree = {}, 0.0, -1
        for b in np.unique(btr):
            m = btr == b; e, v = effect_stats(zt[m], tt[m])
            if np.isfinite(e) and v < np.inf:
                eff[b] = e; Q += (e - e0) ** 2 / v; dfree += 1
        w = max(Q - dfree, 0.0)                     # разница эффектов сверх шума
        if w > 0:
            num += w * np.array([eff.get(b, e0) - e0 for b in bte]); den += w
    return num / den if den > 0 else np.zeros(len(te))

# --- 3b. Дерево: делим там, где разница эффектов больше всего ---
def best_split(rows, feats, min_arm, z_crit):
    zt, tt = Z[rows], t[rows]; best = None
    for j in feats:
        x = X[rows, j]
        for c in thresholds(x):
            left = x <= c                            # NaN уходят вправо
            nl1, nl0 = (left & (tt == 1)).sum(), (left & (tt == 0)).sum()
            nr1, nr0 = (~left & (tt == 1)).sum(), (~left & (tt == 0)).sum()
            if min(nl1, nl0, nr1, nr0) < min_arm: continue
            el, vl = effect_stats(zt[left], tt[left]); er, vr = effect_stats(zt[~left], tt[~left])
            tstat = abs(el - er) / np.sqrt(vl + vr)
            if tstat > z_crit and (best is None or tstat > best[0]): best = (tstat, j, c)
    return best

def grow_tree(rows, max_leaves, min_arm, feats=None, z_crit=None, rng=None, max_depth=4):
    feats = range(X.shape[1]) if feats is None else feats
    z_crit = z_crit if z_crit is not None else 3.5                 # строго: перебираем сотни разрезов
    leaves = [dict(rows=rows, rules=[], depth=0)]
    while len(leaves) < max_leaves:
        cands = []
        for i, lf in enumerate(leaves):
            if lf['depth'] >= max_depth: continue
            fs = feats if rng is None else rng.choice(X.shape[1], int(np.sqrt(X.shape[1])) + 1, replace=False)
            b = best_split(lf['rows'], fs, min_arm, z_crit)
            if b: cands.append((b[0], i, b[1], b[2]))
        if not cands: break
        _, i, j, c = max(cands)
        lf = leaves.pop(i); x = X[lf['rows'], j]; left = x <= c
        leaves += [dict(rows=lf['rows'][left], rules=lf['rules'] + [(j, '<=', c)], depth=lf['depth'] + 1),
                   dict(rows=lf['rows'][~left], rules=lf['rules'] + [(j, '>', c)], depth=lf['depth'] + 1)]
    return [lf['rules'] for lf in leaves]

def leaf_of(rules_list, rows):
    out = np.full(len(rows), -1)
    for k, rules in enumerate(rules_list):
        m = np.ones(len(rows), bool)
        for j, op, c in rules:
            x = X[rows, j]; m &= (x <= c) if op == '<=' else ~(x <= c)
        out[m] = k
    return out

def tree_score(tr, te, rng, max_leaves=MAX_LEAVES, min_arm=MIN_ARM, return_tree=False):
    half = rng.random(len(tr)) < .5; A, E = tr[half], tr[~half]          # честно: структура на A, эффекты на E
    rules = grow_tree(A, max_leaves, min_arm // 2, feats=FT)
    lE, lT = leaf_of(rules, E), leaf_of(rules, te)
    eff = np.array([effect_stats(Z[E][lE == k], t[E][lE == k])[0] for k in range(len(rules))])
    score = np.nan_to_num(eff[lT])
    return (score, rules) if return_tree else score

def forest_score(tr, te, rng, n_trees=N_TREES):
    s = np.zeros(len(te))
    for _ in range(n_trees):
        sub = rng.choice(tr, len(tr) // 2, replace=False); half = rng.random(len(sub)) < .5
        rules = grow_tree(sub[half], 16, 200, z_crit=1.0, rng=rng, max_depth=5)
        E = sub[~half]; lE, lT = leaf_of(rules, E), leaf_of(rules, te)
        eff = np.array([effect_stats(Z[E][lE == k], t[E][lE == k])[0] for k in range(len(rules))])
        s += np.nan_to_num(eff[lT])
    return s / n_trees

# =====================================================================================
# 4. Известные методы (настроенные) на тех же признаках
# =====================================================================================
GRID = [dict(max_iter=150, learning_rate=0.1, max_depth=3), dict(max_iter=300, learning_rate=0.05, max_depth=5),
        dict(max_iter=400, learning_rate=0.03, max_depth=None, min_samples_leaf=300, l2_regularization=1.0)]
def fit_methods(Xa, tt, yy, Xt, prm):
    out = {}; Cm = lambda: C(**prm); Rm = lambda: R(**prm)
    s = Cm().fit(np.c_[Xa, tt], yy)
    out['S-learner'] = s.predict_proba(np.c_[Xt, np.ones(len(Xt))])[:, 1] - s.predict_proba(np.c_[Xt, np.zeros(len(Xt))])[:, 1]
    m1, m0 = Cm().fit(Xa[tt == 1], yy[tt == 1]), Cm().fit(Xa[tt == 0], yy[tt == 0])
    out['T-learner'] = m1.predict_proba(Xt)[:, 1] - m0.predict_proba(Xt)[:, 1]
    p = tt.mean(); p1, p0 = m1.predict_proba(Xa)[:, 1], m0.predict_proba(Xa)[:, 1]
    d1 = Rm().fit(Xa[tt == 1], yy[tt == 1] - p0[tt == 1]); d0 = Rm().fit(Xa[tt == 0], p1[tt == 0] - yy[tt == 0])
    out['X-learner'] = p * d0.predict(Xt) + (1 - p) * d1.predict(Xt)
    out['DR-learner'] = Rm().fit(Xa, p1 - p0 + tt * (yy - p1) / p - (1 - tt) * (yy - p0) / (1 - p)).predict(Xt)
    out['Class transformation'] = 2 * Cm().fit(Xa, (tt == yy).astype(int)).predict_proba(Xt)[:, 1] - 1
    return out
try:
    from econml.dml import CausalForestDML; HAS_CF = True
except ImportError:
    HAS_CF = False; print('econml не установлен — Causal Forest пропущен')
def learners(tr, te, rng):
    Xa, Xt, tt, yy = X[tr], X[te], t[tr], y[tr]; v = rng.random(len(tr)) < .25; best = {}
    for prm in GRID:
        for m, sc in fit_methods(Xa[~v], tt[~v], yy[~v], Xa[v], prm).items():
            q = qini(yy[v], tt[v], sc)
            if m not in best or q > best[m][0]: best[m] = (q, prm)
    out = {}
    for prm in {str(p): p for _, p in best.values()}.values():
        out.update({m + ' (настр.)': sc for m, sc in fit_methods(Xa, tt, yy, Xt, prm).items() if best[m][1] == prm})
    if USE_CAUSAL_FOREST and HAS_CF:
        Xi, Xti = np.nan_to_num(Xa, nan=-1), np.nan_to_num(Xt, nan=-1)
        cf = CausalForestDML(model_y=R(max_iter=200), model_t=C(max_iter=100), discrete_treatment=True,
                             n_estimators=100, min_samples_leaf=100, random_state=0)
        cf.fit(yy, tt, X=Xi); out['Causal Forest'] = cf.effect(Xti)
    return out, out.get('S-learner (настр.)')

def distilled_tree(tr, te, s_tr):
    """Стандартный способ дать маркетологу сегменты из ML: дерево на 8 листьев поверх прогнозов S-learner."""
    dt = DecisionTreeRegressor(max_leaf_nodes=MAX_LEAVES, min_samples_leaf=2 * MIN_ARM)
    dt.fit(np.nan_to_num(X[tr], nan=-1), s_tr); return dt.predict(np.nan_to_num(X[te], nan=-1))

# =====================================================================================
# 5. Сравнение
# =====================================================================================
print('\n=== Часть 1. Ранжирование (все клиенты в обучении, тест — отложенные 30%) ===')
rows = []
for seed in range(SEEDS):
    rng = np.random.default_rng(seed); perm = rng.permutation(N); te, tr = perm[: int(N * .3)], perm[int(N * .3):]
    ml, _ = learners(tr, te, rng)
    prm = GRID[1]; s_model = C(**prm).fit(np.c_[X[tr], t[tr]], y[tr])
    s_tr = s_model.predict_proba(np.c_[X[tr], np.ones(len(tr))])[:, 1] - s_model.predict_proba(np.c_[X[tr], np.zeros(len(tr))])[:, 1]
    preds = {'Ваш: сегменты': segments_score(tr, te), 'Ваш: дерево (≤8 сегментов)': tree_score(tr, te, rng),
             'Ваш: лес': forest_score(tr, te, rng), 'ML-дерево (≤8 сегментов)': distilled_tree(tr, te, s_tr),
             **ml, 'Рандом': rng.random(len(te))}
    for m, sc in preds.items():
        rows.append(dict(метод=m, seed=seed, qini=qini(y[te], t[te], sc), uplift30=uplift_at30(y[te], t[te], sc)))
    print(f'  повтор {seed + 1}/{SEEDS}, {time.time() - t0:.0f} c', flush=True)
res = pd.DataFrame(rows); res.to_csv('x5_v3_results.csv', index=False)
tab = res.groupby('метод').agg(qini=('qini', 'mean'), qini_sd=('qini', 'std'), uplift30=('uplift30', 'mean'),
                               uplift30_sd=('uplift30', 'std')).sort_values('uplift30', ascending=False)
pd.set_option('display.width', 250); print(tab.round(4).to_string())
print('\nОриентиры соревнования (uplift@30%, скрытый тест): публичный ~0.099–0.106, приватный 4-е место 0.0983')

def compare(a, b):
    w = res.pivot_table(index='seed', columns='метод', values='uplift30'); d = w[a] - w[b]
    bs = [d.sample(len(d), replace=True, random_state=i).mean() for i in range(2000)]
    print(f'  {a} − {b}: {d.mean():+.4f} (95%: {np.quantile(bs, .025):+.4f}…{np.quantile(bs, .975):+.4f}), побед {(d > 0).sum()} из {len(d)}')
mlnames = [m for m in res.метод.unique() if m.endswith('(настр.)') or m == 'Causal Forest']
best_ml = res[res.метод.isin(mlnames)].groupby('метод').uplift30.mean().idxmax()
print('\nСравнение (uplift@30%):')
for mine in ['Ваш: лес', 'Ваш: сегменты']: compare(mine, best_ml)
compare('Ваш: дерево (≤8 сегментов)', 'ML-дерево (≤8 сегментов)')

# =====================================================================================
# 6. Отчёт для маркетолога: сегменты дерева, проверенные на независимых данных
# =====================================================================================
print('\n=== Часть 2. Сегменты для маркетолога ===')
rng = np.random.default_rng(100); perm = rng.permutation(N); A, B = perm[: N // 2], perm[N // 2:]
rules = grow_tree(A, MAX_LEAVES, MIN_ARM, feats=FT)
def rule_text(rules_):
    parts = {}
    for j, op, c in rules_:                       # схлопываем повторы по одной фиче в диапазон
        lo, hi = parts.get(j, (-np.inf, np.inf))
        parts[j] = (lo, min(hi, c)) if op == '<=' else (max(lo, c), hi)
    out = []
    for j, (lo, hi) in parts.items():
        nm = FEATS[j]
        if np.isfinite(lo) and np.isfinite(hi): out.append(f'{lo:.3g} < {nm} ≤ {hi:.3g}')
        elif np.isfinite(hi): out.append(f'{nm} ≤ {hi:.3g}')
        else: out.append(f'{nm} > {lo:.3g}')
    return ' и '.join(out) or 'все клиенты'
lB = leaf_of(rules, B); rep = []
for k, r_ in enumerate(rules):
    rows_ = B[lB == k]; sims = mc_effect(rows_, rng)
    m, c = rows_[t[rows_] == 1], rows_[t[rows_] == 0]; raw = y[m].mean() - y[c].mean()
    lo, hi = np.quantile(sims, [.025, .975])
    rep.append(dict(сегмент=rule_text(r_), доля_клиентов=len(rows_) / len(B), эффект_пп=100 * sims.mean(),
                    низ_пп=100 * lo, верх_пп=100 * hi, простая_разница_пп=100 * raw,
                    доп_покупателей_на_1000_SMS=1000 * sims.mean(),
                    прибыль_на_1000_SMS=1000 * sims.mean() * PROFIT_PER_BUYER - 1000 * SMS_COST))
rep = pd.DataFrame(rep).sort_values('эффект_пп', ascending=False)
allB = mc_effect(B, rng)
print(f'Все клиенты (половина B): {100 * allB.mean():+.2f} п.п. ({100 * np.quantile(allB, .025):+.2f}…{100 * np.quantile(allB, .975):+.2f})')
print(f'Дерево построено на половине A, эффекты посчитаны на независимой половине B. Сегментов: {len(rules)}')
print(rep.round(2).to_string(index=False))
print(f'\nЦены: SMS {SMS_COST} руб., прибыль с доп. покупателя {PROFIT_PER_BUYER} руб. Прибыль < 0 — не окупается.')
print(f'Всего времени: {time.time() - t0:.0f} c')
