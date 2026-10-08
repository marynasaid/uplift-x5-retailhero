"""X5 RetailHero, версия 2: метод «мешочков + Монте-Карло» против НАСТРОЕННЫХ моделей и Causal Forest,
метрика соревнования uplift@30%, эффект сегментов в п.п. и деньгах.

В Colab:
    from google.colab import drive; drive.mount('/content/drive')
    %run /content/drive/MyDrive/ПУТЬ/x5_method.py
Поправьте DATA ниже: там должны лежать purchases.csv, clients.csv, uplift_train.csv (можно .csv.gz).
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as R, HistGradientBoostingClassifier as C
warnings.filterwarnings('ignore')

DATA = '/content/drive/MyDrive/x5'          # <-- папка с файлами
WINDOW_DAYS = 7                             # длина окна
TRAIN_SIZES = [3_000, 10_000, None]         # None = все клиенты
SEEDS = 10                                  # повторов (20 — надёжнее, но вдвое дольше)
USE_CAUSAL_FOREST = True                    # нужен !pip install econml (иначе пропускается)
SMS_COST = 2.0                              # стоимость одной SMS, руб.   <-- поставьте свои
PROFIT_PER_BUYER = 150.0                    # прибыль с одного доп. покупателя, руб.  <-- свои
N_MC = 200                                   # итераций Монте-Карло

def path(name):
    for ext in ['', '.gz']:
        p = os.path.join(DATA, name + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f'{name} не найден в {DATA}')

# =====================================================================================
# 1. Окна и мешочки
# =====================================================================================
t0 = time.time()
train = pd.read_csv(path('uplift_train.csv'))
clients = pd.read_csv(path('clients.csv')).set_index('client_id').loc[train.client_id]
ids = set(train.client_id)
parts = []
for ch in pd.read_csv(path('purchases.csv'), chunksize=2_000_000,
                      usecols=['client_id', 'transaction_id', 'transaction_datetime', 'purchase_sum']):
    parts.append(ch[ch.client_id.isin(ids)].drop_duplicates('transaction_id'))
tx = pd.concat(parts).drop_duplicates('transaction_id')
tx['dt'] = pd.to_datetime(tx.transaction_datetime)
print(f'Покупки загружены: {len(tx):,} чеков, {time.time() - t0:.0f} c')

start = tx.dt.min().normalize(); end = tx.dt.max().normalize() + pd.Timedelta(days=1)
K = int((end - start).days // WINDOW_DAYS)                 # только полные окна
first = end - pd.Timedelta(days=K * WINDOW_DAYS)           # неполное окно в начале отрезаем
tx = tx[tx.dt >= first]
tx['w'] = ((tx.dt - first).dt.days // WINDOW_DAYS).astype(int)
cidx = pd.Series(np.arange(len(train)), index=train.client_id)
bought = np.zeros((len(train), K), np.float32)             # мешочек: купил ли в окне (1/0)
bought[tx.client_id.map(cidx).values, tx.w.values] = 1
# окна до появления клиента = NaN
issue = pd.to_datetime(clients.first_issue_date).values
arrive = np.clip(((issue - np.datetime64(first)) / np.timedelta64(WINDOW_DAYS, 'D')).astype(float), 0, None)
arrive = np.ceil(arrive).astype(int)
H = np.where(np.arange(K)[None, :] >= arrive[:, None], bought, np.nan)
L = (~np.isnan(H)).sum(1)
keep = L >= 2                                               # хотя бы 2 окна в мешочке
print(f'Окон: {K} по {WINDOW_DAYS} дн. с {first.date()} по {end.date()}; клиентов с ≥2 окнами: {keep.sum():,} из {len(keep):,}')

t = train.treatment_flg.values; y = train.target.values.astype(float)
# --- признаки ---
age = clients.age.where(clients.age.between(14, 100)).values
tenure = (end - pd.to_datetime(clients.first_issue_date)).dt.days.values
redeemed = clients.first_redeem_date.notna().astype(int).values
gender = clients.gender.map({'F': 0, 'M': 1, 'U': 2}).values
act = np.nanmean(H, 1)
last_buy = np.where(np.nan_to_num(H) > 0, np.arange(K)[None, :], -1).max(1)
recency = np.where(last_buy < 0, K, K - 1 - last_buy)       # окон с последней покупки
F = pd.DataFrame(dict(gender=gender, age=age, tenure=tenure, redeemed=redeemed, act=act, recency=recency, L=L))
H, t, y, F = H[keep], t[keep], y[keep], F[keep].reset_index(drop=True)
N = len(y); KW = H.shape[1]
print(f'Treatment {t.mean():.2f}, купили после SMS: main {y[t == 1].mean():.3f}, control {y[t == 0].mean():.3f}')

# --- сегменты (заранее зафиксированы) ---
FEATS = {'пол': ['жен', 'муж', 'не указан'],
         'возраст': ['<30', '30-44', '45-59', '60+', 'не указан'],
         'стаж карты': ['новые', 'средние', 'старые'],
         'бонусы списывал': ['нет', 'да'],
         'активность': ['низкая', 'ниже ср', 'выше ср', 'высокая'],
         'давность': ['0 окон', '1-2 окна', '3+ окон']}
def segments(Fd, q):
    return pd.DataFrame({
        'пол': Fd.gender.values,
        'возраст': np.where(np.isnan(Fd.age), 4, np.searchsorted([30, 45, 60], np.nan_to_num(Fd.age), side='right')),
        'стаж карты': np.searchsorted(q['ten'], Fd.tenure.values),
        'бонусы списывал': Fd.redeemed.values,
        'активность': np.searchsorted(q['act'], Fd.act.values),
        'давность': np.searchsorted([0, 2], Fd.recency.values, side='right')})
def quantiles(Fd):
    return {'ten': np.quantile(Fd.tenure, [1/3, 2/3]), 'act': np.unique(np.quantile(Fd.act, [.25, .5, .75]))}
Q = quantiles(F); S_all = segments(F, Q)

# =====================================================================================
# 2. Монте-Карло: эффект набора клиентов (бутстрэп + вытаскивание из мешочка)
# =====================================================================================
def draw(Hs, Ls, b, rng):
    """U ~ [0,1) → элемент мешочка с индексом floor(U·len) (обратное преобразование)."""
    return Hs[b, KW - Ls[b] + (rng.random(b.shape) * Ls[b]).astype(int)]

def mc_effect(rows, rng, n_mc=N_MC, min_arm=200, variant='отношение'):
    m, c = rows[t[rows] == 1], rows[t[rows] == 0]
    if min(len(m), len(c)) < min_arm: return None
    out = []
    for ids_ in (m, c):
        Hs, Ls, ys = H[ids_], L_[ids_], y[ids_]
        b = rng.integers(0, len(ids_), (n_mc, len(ids_)), dtype=np.int32)
        d = draw(Hs, Ls, b, rng)
        out.append((ys[b].sum(1), d.sum(1), len(ids_)))
    (ym, dm, nm), (yc, dc, nc) = out
    if variant == 'отношение': return ((ym + 1) / (dm + 1)) / ((yc + 1) / (dc + 1))
    return (ym - dm) / nm - (yc - dc) / nc
L_ = F.L.values

# =====================================================================================
# 3. Сравнение с известными методами (Qini)
# =====================================================================================
FEAT_ML = ['gender', 'age', 'tenure', 'redeemed', 'act', 'recency', 'L']

def qini(yv, tv, s):
    o = np.argsort(-s, kind='stable'); yv, tv = yv[o], tv[o]
    nt, nc = np.cumsum(tv), np.cumsum(1 - tv)
    q = np.cumsum(yv * tv) - np.cumsum(yv * (1 - tv)) * np.divide(nt, nc, out=np.zeros(len(nt)), where=nc > 0)
    return (q - np.linspace(0, q[-1], len(q))).mean() / len(q) * 1000

def uplift_at30(yv, tv, s, k=0.3):
    """Метрика соревнования: топ-30% по скору, доля купивших в main минус в control."""
    top = np.argsort(-s, kind='stable')[: int(len(s) * k)]
    return yv[top][tv[top] == 1].mean() - yv[top][tv[top] == 0].mean()

def my_method_scores(tr, te, rng, n_mc=100):
    """Эффект каждого сегмента каждой фичи на train → для клиента test среднее эффектов его сегментов."""
    q = quantiles(F.iloc[tr]); Str, Ste = segments(F.iloc[tr], q), segments(F.iloc[te], q)
    lr, df = [], []
    for f in FEATS:
        e_r, e_d = np.zeros(len(FEATS[f])), np.zeros(len(FEATS[f]))
        for v in range(len(FEATS[f])):
            rows = tr[Str[f].values == v]
            r = mc_effect(rows, rng, n_mc, min_arm=20)
            d = mc_effect(rows, rng, n_mc, min_arm=20, variant='разница')
            e_r[v] = np.log(r.mean()) if r is not None else 0.0
            e_d[v] = d.mean() if d is not None else 0.0
        lr.append(e_r[Ste[f].values]); df.append(e_d[Ste[f].values])
    lr, df = np.mean(lr, 0), np.mean(df, 0)
    return {'Ваш (отношение)': F.act.values[te] * (np.exp(lr) - 1), 'Ваш (разница)': df}

GRID = [dict(max_iter=100, learning_rate=0.1, max_depth=3), dict(max_iter=300, learning_rate=0.05, max_depth=4),
        dict(max_iter=200, learning_rate=0.05, max_depth=None, min_samples_leaf=200),
        dict(max_iter=500, learning_rate=0.02, max_depth=6, l2_regularization=1.0)]

def fit_methods(X, tt, yy, Xt, prm):
    """Все известные методы с гиперпараметрами prm → скоры uplift для Xt."""
    Cm = lambda: C(**prm); Rm = lambda: R(**prm); out = {}
    s = Cm().fit(np.c_[X, tt], yy)
    out['S-learner'] = s.predict_proba(np.c_[Xt, np.ones(len(Xt))])[:, 1] - s.predict_proba(np.c_[Xt, np.zeros(len(Xt))])[:, 1]
    m1, m0 = Cm().fit(X[tt == 1], yy[tt == 1]), Cm().fit(X[tt == 0], yy[tt == 0])
    p1, p0 = (lambda z: m1.predict_proba(z)[:, 1]), (lambda z: m0.predict_proba(z)[:, 1])
    out['T-learner'] = p1(Xt) - p0(Xt)
    e = np.full(len(X), tt.mean()); et = np.full(len(Xt), tt.mean())          # рассылка случайная: p = доля main
    d1 = Rm().fit(X[tt == 1], yy[tt == 1] - p0(X[tt == 1])); d0 = Rm().fit(X[tt == 0], p1(X[tt == 0]) - yy[tt == 0])
    out['X-learner'] = et * d0.predict(Xt) + (1 - et) * d1.predict(Xt)
    mu1, mu0 = p1(X), p0(X)
    out['DR-learner'] = Rm().fit(X, mu1 - mu0 + tt * (yy - mu1) / e - (1 - tt) * (yy - mu0) / (1 - e)).predict(Xt)
    z = (tt == yy).astype(int); out['Class transformation'] = 2 * Cm().fit(X, z).predict_proba(Xt)[:, 1] - 1
    return out

def learners(tr, te, rng):
    """Настройка: для каждого метода выбираем лучшие параметры по Qini на 25% train, потом учим на всём train."""
    X, Xt, tt, yy = F.iloc[tr][FEAT_ML].values, F.iloc[te][FEAT_ML].values, t[tr], y[tr]
    v = rng.random(len(tr)) < .25
    best = {}
    for prm in GRID:
        for m, sc in fit_methods(X[~v], tt[~v], yy[~v], X[v], prm).items():
            q = qini(yy[v], tt[v], sc)
            if m not in best or q > best[m][0]: best[m] = (q, prm)
    out = {}
    for prm in {str(p): p for _, p in best.values()}.values():
        res = fit_methods(X, tt, yy, Xt, prm)
        out.update({m + ' (настр.)': sc for m, sc in res.items() if best[m][1] == prm})
    if USE_CAUSAL_FOREST and HAS_CF:
        cf = CausalForestDML(model_y=R(max_iter=200), model_t=C(max_iter=100), discrete_treatment=True,
                             n_estimators=200, min_samples_leaf=50, random_state=0)
        cf.fit(yy.astype(float), tt, X=X); out['Causal Forest'] = cf.effect(Xt)
    return out

try:
    from econml.dml import CausalForestDML; HAS_CF = True
except ImportError:
    HAS_CF = False
    if USE_CAUSAL_FOREST: print('econml не установлен — Causal Forest пропущен (в Colab: !pip install econml)')

print('\n=== Часть 1. Qini против известных методов (больше = лучше, 0 = случайный порядок) ===')
rows = []
for n_tr in TRAIN_SIZES:
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed); perm = rng.permutation(N)
        te = perm[: int(N * .3)]; pool = perm[int(N * .3):]
        tr = pool if n_tr is None else pool[:n_tr]
        preds = {**my_method_scores(tr, te, rng), **learners(tr, te, rng), 'Рандом': rng.random(len(te))}
        for m, sc in preds.items():
            rows.append(dict(клиентов=len(tr), метод=m, seed=seed, qini=qini(y[te], t[te], sc),
                             uplift30=uplift_at30(y[te], t[te], sc)))
        print(f'    повтор {seed + 1}/{SEEDS}', flush=True)
    print(f'  обучение на {len(tr):,} клиентах готово', flush=True)
res = pd.DataFrame(rows); res.to_csv('x5_part1_results.csv', index=False)
pd.set_option('display.width', 250)
for metric in ['qini', 'uplift30']:
    print(f'\n--- {metric} (среднее по {SEEDS} повторам) ---')
    print(res.pivot_table(index='клиентов', columns='метод', values=metric).round(4).to_string())
ml = [m for m in res.метод.unique() if m.endswith('(настр.)') or m == 'Causal Forest']
print('\nВаш (разница) против лучшего настроенного метода:')
for n, g in res.groupby('клиентов'):
    for metric in ['qini', 'uplift30']:
        w = g.pivot_table(index='seed', columns='метод', values=metric)
        best = w[ml].mean().idxmax(); d = w['Ваш (разница)'] - w[best]
        lo, hi = np.quantile([d.sample(len(d), replace=True, random_state=i).mean() for i in range(2000)], [.025, .975])
        print(f'  {n:>7,} клиентов, {metric:8s}: против {best:28s} {(d > 0).sum():2d} побед из {len(d)}, '
              f'средняя разница {d.mean():+.4f} (95%: {lo:+.4f}…{hi:+.4f})')

full = res[res.клиентов == res.клиентов.max()].groupby('метод').uplift30.mean().sort_values(ascending=False)
print('\n=== Сравнение с соревнованием RetailHero (метрика uplift@30%) ===')
print('Ориентиры участников (на скрытом тесте соревнования, с ~150 сконструированными признаками):')
print('  публичный лидерборд: ~0.099–0.106;  приватный, 4-е место: 0.0983')
print('Наши результаты (отложенные 30% train, все клиенты в обучении):')
print(full.round(4).to_string())
print('Важно: тест другой (у нас — часть train), поэтому сравнение ориентировочное.')

# =====================================================================================
# 4. Поиск сегментов вглубь: половина A — поиск, половина B — проверка
# =====================================================================================
CONF = {1: 0.95, 2: 0.99, 3: 0.999}; BAND = (0.95, 1.05); MAX_DEPTH = 3; VARIANT = 'отношение'
def seg_rows(rows, cond):
    for f, v in cond: rows = rows[S_all[f].values[rows] == v]
    return rows
def classify(sims, level):
    a = (1 - CONF[level]) / 2; lo, hi = np.quantile(sims, [a, 1 - a]); base = 1.0 if VARIANT == 'отношение' else 0.0
    if lo > base or hi < base: return 'есть', sims.mean(), lo, hi
    band = BAND if VARIANT == 'отношение' else (-0.01, 0.01)
    if band[0] <= lo and hi <= band[1]: return 'нет', sims.mean(), lo, hi
    return 'непонятно', sims.mean(), lo, hi
def name(cond): return ' & '.join(f'{f}={FEATS[f][v]}' for f, v in cond)
def eff(rows, rng, cache, cond):
    k = tuple(sorted(cond))
    if k not in cache: cache[k] = mc_effect(rows, rng, variant=VARIANT)
    return cache[k]
def stronger(child, parent, level):
    a = (1 - CONF[min(level, MAX_DEPTH)]) / 2
    up = parent.mean() >= (1.0 if VARIANT == 'отношение' else 0.0)
    return np.quantile(child, a) > parent.mean() if up else np.quantile(child, 1 - a) < parent.mean()
def refine(rows, rng, cond, sims, level, cache):
    if level >= MAX_DEPTH: return cond
    best = None
    for f in FEATS:
        if f in {g for g, _ in cond}: continue
        for v in range(len(FEATS[f])):
            sub = rows[S_all[f].values[rows] == v]; new = cond + ((f, v),); s = eff(sub, rng, cache, new)
            if s is not None and stronger(s, sims, level + 1) and (best is None or abs(np.log(abs(s.mean()) + 1e-9)) > abs(np.log(abs(best[1].mean()) + 1e-9))):
                best = (new, s, sub)
    return cond if best is None else refine(best[2], rng, best[0], best[1], level + 1, cache)
def search(rows, rng, cond=(), level=1, cache=None, log=None):
    found = []
    for f in FEATS:
        if f in {g for g, _ in cond}: continue
        for v in range(len(FEATS[f])):
            sub = rows[S_all[f].values[rows] == v]; new = cond + ((f, v),); s = eff(sub, rng, cache, new)
            if s is None: continue
            st, e, lo, hi = classify(s, level); log.append((level, name(new), st, e, lo, hi))
            if st == 'есть': found.append(refine(sub, rng, new, s, level, cache))
            elif st == 'нет' and level < MAX_DEPTH: found += search(sub, rng, new, level + 1, cache, log)
    return list({tuple(sorted(c)): c for c in found}.values())
def cleanup(rows, cands, rng):
    stats = {c: mc_effect(seg_rows(rows, c), rng, variant=VARIANT) for c in cands}
    stats = {c: s for c, s in stats.items() if s is not None}
    drop = {ch for ch in stats for pa in stats if pa != ch and set(pa) < set(ch) and not stronger(stats[ch], stats[pa], len(ch))}
    order = sorted([c for c in stats if c not in drop], key=lambda c: -abs(np.log(abs(stats[c].mean()) + 1e-9)))
    kept, covered = [], np.zeros(N, bool)
    for c in order:
        r = seg_rows(rows, c); s = mc_effect(r[~covered[r]], rng, variant=VARIANT)
        if s is not None and classify(s, len(c))[0] == 'есть': kept.append(c); covered[r] = True
    return kept

print(f'\n=== Часть 2. Поиск сегментов (эффект: {VARIANT}) ===')
rng = np.random.default_rng(42); perm = rng.permutation(N); A, B = perm[: N // 2], perm[N // 2:]
allA = mc_effect(A, rng, variant=VARIANT); allB = mc_effect(B, rng, variant=VARIANT)
print(f'Все клиенты: A ×{allA.mean():.3f} ({np.quantile(allA, .025):.3f}–{np.quantile(allA, .975):.3f}), '
      f'B ×{allB.mean():.3f} ({np.quantile(allB, .025):.3f}–{np.quantile(allB, .975):.3f})')
log, cache = [], {}
raw = search(A, rng, cache=cache, log=log)
print('\nПервый уровень на половине A:')
for lvl, nm, st, e, lo, hi in log:
    if lvl == 1: print(f'  {nm:35s} ×{e:.3f} ({lo:.3f}–{hi:.3f}) → {st}')
cands = cleanup(A, raw, rng)
print(f'\nНайдено на A: {len(raw)}, после чистки: {len(cands)}')
report, covB = [], np.zeros(N, bool)
for c in cands:
    rB = seg_rows(B, c); rB_free = rB[~covB[rB]]; covB[rB] = True
    sA = mc_effect(seg_rows(A, c), rng, variant=VARIANT); sB = mc_effect(rB_free, rng, variant=VARIANT)
    if sB is None: continue
    loB, hiB = np.quantile(sB, [.025, .975]); base = 1.0 if VARIANT == 'отношение' else 0.0
    report.append(dict(сегмент=name(c), эффект_A=sA.mean(), эффект_B=sB.mean(), B_низ=loB, B_верх=hiB,
                       клиентов_B=len(rB_free), подтверждён=(loB > base) or (hiB < base)))
print('\nОтчёт (проверка на половине B):')
print(pd.DataFrame(report).round(3).to_string(index=False) if report else '  сегментов не найдено')

# =====================================================================================
# 5. Эффект в п.п. и деньгах (на половине B: рассылка случайная, поэтому разница долей честная)
# =====================================================================================
def money(rows, label):
    m, c = rows[t[rows] == 1], rows[t[rows] == 0]
    pm, pc = y[m].mean(), y[c].mean(); d = pm - pc
    se = np.sqrt(pm * (1 - pm) / len(m) + pc * (1 - pc) / len(c))
    extra = 1000 * d; profit = extra * PROFIT_PER_BUYER - 1000 * SMS_COST
    return dict(сегмент=label, клиентов=len(rows), купили_main=pm, купили_control=pc, эффект_пп=100 * d,
                пп_низ=100 * (d - 1.96 * se), пп_верх=100 * (d + 1.96 * se),
                доп_покупателей_на_1000_SMS=extra, прибыль_на_1000_SMS_руб=profit)
mrows = [money(B, 'все клиенты')]
for c in [r['сегмент'] for r in report if r['подтверждён']]:
    cond = next(cc for cc in cands if name(cc) == c); mrows.append(money(seg_rows(B, cond), c))
mrows.append(money(B[S_all['активность'].values[B] == 3], 'активность=высокая (для сравнения)'))
print(f'\n=== Эффект в деньгах (SMS {SMS_COST} руб., прибыль с доп. покупателя {PROFIT_PER_BUYER} руб.) ===')
print(pd.DataFrame(mrows).round(2).to_string(index=False))
print('Прибыль < 0 — рассылка в этом сегменте не окупается при заданных ценах.')
print(f'\nВсего времени: {time.time() - t0:.0f} c')
