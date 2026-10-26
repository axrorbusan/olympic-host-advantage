"""
Do host countries win more Olympic medals?  (1896–2016, athlete-level data)

Questions
  1. How much bigger is a country's medal share in the Games it hosts, compared with its own
     Games before and after?
  2. Is the host effect the same in Summer and Winter?
  3. How much of it is simply sending a bigger team?

Data trap: the file has one row per *athlete*, so a basketball gold appears 12 times.
Medals are counted once per (Games, event, country, medal).

Input : data/raw/olympics.csv.gz (TidyTuesday 2024-08-06, from Sports Reference)
Output: output/*.png, output/results.json, output/host_table.csv
Run   : python analyze.py
"""
import json
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
INK, MUTED, ACCENT, BLUE = "#1F2933", "#6B7280", "#C2410C", "#1D4ED8"

HOSTS = {
    "Summer": {1896: "GRE", 1900: "FRA", 1904: "USA", 1906: "GRE", 1908: "GBR", 1912: "SWE", 1920: "BEL",
               1924: "FRA", 1928: "NED", 1932: "USA", 1936: "GER", 1948: "GBR", 1952: "FIN", 1956: "AUS",
               1960: "ITA", 1964: "JPN", 1968: "MEX", 1972: "FRG", 1976: "CAN", 1980: "URS", 1984: "USA",
               1988: "KOR", 1992: "ESP", 1996: "USA", 2000: "AUS", 2004: "GRE", 2008: "CHN", 2012: "GBR",
               2016: "BRA"},
    "Winter": {1924: "FRA", 1928: "SUI", 1932: "USA", 1936: "GER", 1948: "SUI", 1952: "NOR", 1956: "ITA",
               1960: "USA", 1964: "AUT", 1968: "FRA", 1972: "JPN", 1976: "AUT", 1980: "USA", 1984: "YUG",
               1988: "CAN", 1992: "FRA", 1994: "NOR", 1998: "JPN", 2002: "USA", 2006: "ITA", 2010: "CAN",
               2014: "RUS"},
}
BOYCOTT = {1980: "US-led boycott", 1984: "Soviet-led boycott"}   # Summer only


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titleweight": "bold", "axes.titlesize": 13, "font.size": 10})


def main():
    style()
    os.makedirs(P("output"), exist_ok=True)
    o = pd.read_csv(P("data", "raw", "olympics.csv.gz"))
    res = {"athlete_rows": int(len(o)), "games": int(o.games.nunique())}

    # ---------------------------------------------------------------- the counting trap
    m = o.dropna(subset=["medal"])
    u = m.drop_duplicates(["games", "event", "noc", "medal"])
    res["medal_rows_naive"] = int(len(m))
    res["medals_corrected"] = int(len(u))
    checks = {("2008 Summer", "CHN"): 100, ("2012 Summer", "GBR"): 65, ("1980 Summer", "URS"): 195,
              ("2016 Summer", "USA"): 121, ("2014 Winter", "RUS"): 33, ("1984 Summer", "USA"): 174}
    res["validation_vs_official_totals"] = {
        f"{g} {n}": {"naive": int(((m.games == g) & (m.noc == n)).sum()),
                     "corrected": int(((u.games == g) & (u.noc == n)).sum()), "official": off}
        for (g, n), off in checks.items()}

    # ---------------------------------------------------------------- panel: country × Games
    ath = o.groupby(["games", "year", "season", "noc"])["id"].nunique().rename("athletes")
    med = u.groupby(["games", "noc"]).size().rename("medals")
    pan = ath.reset_index().merge(med.reset_index(), on=["games", "noc"], how="left").fillna({"medals": 0})
    pan["total_medals"] = pan.groupby("games")["medals"].transform("sum")
    pan["share"] = pan["medals"] / pan["total_medals"]
    pan["host"] = [int(HOSTS[s].get(y) == n) for s, y, n in zip(pan.season, pan.year, pan.noc)]
    missing = [(s, y, n) for s, d in HOSTS.items() for y, n in d.items()
               if not ((pan.season == s) & (pan.year == y) & (pan.noc == n)).any()]
    res["hosts_missing_in_data"] = missing

    # ---------------------------------------------------------------- 1. before / host / after
    rows = []
    for season, d in HOSTS.items():
        yrs = sorted(pan[pan.season == season].year.unique())
        for y, noc in d.items():
            i = yrs.index(y)
            near = yrs[max(0, i - 2):i] + yrs[i + 1:i + 3]
            cmp = pan[(pan.season == season) & (pan.noc == noc) & pan.year.isin(near)]
            h = pan[(pan.season == season) & (pan.noc == noc) & (pan.year == y)].iloc[0]
            if len(cmp) == 0:
                continue
            rows.append(dict(season=season, year=y, noc=noc, host_share=h.share, host_medals=int(h.medals),
                             host_athletes=int(h.athletes), other_share=cmp.share.mean(),
                             other_athletes=cmp.athletes.mean(), n_compare=len(cmp),
                             boycott=BOYCOTT.get(y) if season == "Summer" else None))
    ht = pd.DataFrame(rows)
    ht["share_ratio"] = ht.host_share / ht.other_share.replace(0, np.nan)
    ht["athlete_ratio"] = ht.host_athletes / ht.other_athletes
    ht.to_csv(P("output", "host_table.csv"), index=False)
    for s in ("Summer", "Winter"):
        h = ht[(ht.season == s) & ht.share_ratio.notna()]
        hb = h[h.boycott.isna()]
        res[f"{s.lower()}_before_after"] = {
            "n_hosts": int(len(h)),
            "share_higher_when_hosting": int((h.host_share > h.other_share).sum()),
            "median_share_ratio": float(h.share_ratio.median()),
            "median_share_ratio_excl_boycott_games": float(hb.share_ratio.median()),
            "median_athlete_ratio": float(h.athlete_ratio.median()),
            "median_extra_share_points": float(100 * (h.host_share - h.other_share).median()),
            "median_share_ratio_since_1948": float(h[h.year >= 1948].share_ratio.median()),
            "median_athlete_ratio_since_1948": float(h[h.year >= 1948].athlete_ratio.median()),
            "n_hosts_since_1948": int((h.year >= 1948).sum()),
        }

    # ---------------------------------------------------------------- 2. Poisson model with FE
    # medals ~ host + country FE + Games FE, offset log(total medals) → multiplier on medal share
    models = {}
    for s in ("Summer", "Winter", "Summer_since_1948"):
        d = pan[(pan.season == s.split("_")[0])].copy()
        if s.endswith("1948"):
            d = d[d.year >= 1948]
        d = d[d.groupby("noc").medals.transform("sum") > 0]           # countries that ever won
        d["log_total"] = np.log(d.total_medals)
        d["log_athletes"] = np.log(d.athletes)
        for name, f in (("host_only", "medals ~ host + C(noc) + C(games)"),
                        ("plus_team_size", "medals ~ host + log_athletes + C(noc) + C(games)")):
            mdl = smf.glm(f, data=d, family=sm.families.Poisson(), offset=d.log_total).fit(
                cov_type="cluster", cov_kwds={"groups": d.noc.astype("category").cat.codes})
            b, se = mdl.params["host"], mdl.bse["host"]
            models[(s, name)] = dict(multiplier=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)),
                                     hi=float(np.exp(b + 1.96 * se)), p=float(mdl.pvalues["host"]),
                                     n_obs=int(mdl.nobs))
            if name == "plus_team_size":
                models[(s, name)]["athlete_elasticity"] = float(mdl.params["log_athletes"])
    res["poisson_fe"] = {f"{s}_{n}": v for (s, n), v in models.items()}
    share_explained = {s: 1 - np.log(models[(s, "plus_team_size")]["multiplier"]) /
                          np.log(models[(s, "host_only")]["multiplier"]) for s in ("Summer", "Winter")}
    res["share_of_host_effect_explained_by_team_size"] = {k: float(v) for k, v in share_explained.items()}

    # ---------------------------------------------------------------- fig 1: before/after vs host (Summer)
    h = ht[ht.season == "Summer"].sort_values("year")
    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(h))[::-1]
    ax.hlines(y, 100 * h.other_share, 100 * h.host_share, color="#D1D5DB", lw=2.5)
    ax.scatter(100 * h.other_share, y, color="#9CA3AF", s=36, zorder=3, label="Average of 2 Games before & 2 after")
    ax.scatter(100 * h.host_share, y, color=ACCENT, s=42, zorder=3, label="As host")
    lab = [f"{r.year} {r.noc}" + (" *" if isinstance(r.boycott, str) else "") for _, r in h.iterrows()]
    ax.set_yticks(y, lab)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5%", "1%", "2%", "5%", "10%", "20%", "50%"])
    ax.set_xlabel("Share of all medals awarded at the Games (log scale)")
    ax.set_title("Summer Olympics: host countries win a bigger share", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.text(0, -0.09, "* boycotted Games (1980, 1984): rivals absent, inflating the host share.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.grid(axis="x", color="#E5E7EB")
    fig.savefig(P("output", "fig1_summer_hosts.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 2: naive vs corrected
    nv = m.groupby("noc").size().rename("naive")
    cr = u.groupby("noc").size().rename("corrected")
    t = pd.concat([nv, cr], axis=1).nlargest(12, "naive")
    t["inflation"] = t.naive / t.corrected
    res["all_time_top12_inflation"] = {k: float(v) for k, v in t.inflation.items()}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.naive, 0.4, color="#D1D5DB", label="Naive: one row per athlete")
    ax.bar(x + 0.2, t.corrected, 0.4, color=ACCENT, label="Corrected: one per event")
    for xi, (_, r) in zip(x, t.iterrows()):
        ax.text(xi, r.naive + 60, f"×{r.inflation:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, t.index)
    ax.set_ylabel("Total medals 1896–2016")
    ax.set_title("Counting athletes instead of medals roughly doubles the totals", loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#E5E7EB")
    fig.savefig(P("output", "fig2_counting_trap.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 3: multipliers
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, ys = [], []
    for i, (s, n, txt) in enumerate([("Summer", "host_only", "Summer, 1896–2016"),
                                     ("Summer", "plus_team_size", "Summer, controlling for team size"),
                                     ("Summer_since_1948", "host_only", "Summer, 1948–2016 only"),
                                     ("Winter", "host_only", "Winter"),
                                     ("Winter", "plus_team_size", "Winter, controlling for team size")]):
        r = models[(s, n)]
        yy = 4 - i
        ax.errorbar(r["multiplier"], yy, xerr=[[r["multiplier"] - r["lo"]], [r["hi"] - r["multiplier"]]],
                    fmt="o", color=(ACCENT if n == "host_only" else BLUE) if "1948" not in s else "#9CA3AF",
                    capsize=3, ms=7)
        ax.text(r["hi"] + 0.05, yy, f"×{r['multiplier']:.2f}", va="center", fontsize=9)
        labels.append(txt); ys.append(yy)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_yticks(ys, labels)
    ax.set_xlabel("Medal-share multiplier when hosting (95% CI)")
    ax.set_title("Host effect: country & Games fixed effects (Poisson)", loc="left")
    ax.grid(axis="x", color="#E5E7EB")
    fig.savefig(P("output", "fig3_host_multiplier.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- fig 4: athletes vs medals ratio
    hh = ht[ht.share_ratio.notna() & (ht.share_ratio > 0)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for s, c in (("Summer", ACCENT), ("Winter", BLUE)):
        d = hh[hh.season == s]
        ax.scatter(d.athlete_ratio, d.share_ratio, color=c, s=40, label=s, alpha=0.85)
        for _, r in d[(d.share_ratio > 3) | (d.share_ratio < 1) | (d.athlete_ratio > 3)].iterrows():
            ax.annotate(f"{r.noc} {r.year}", (r.athlete_ratio, r.share_ratio), xytext=(3, 2),
                        textcoords="offset points", fontsize=7, color=c)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":"); ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Athletes sent as host ÷ average of nearby Games")
    ax.set_ylabel("Medal share as host ÷ average of nearby Games")
    ax.set_title("Hosts send bigger teams — and win more", loc="left")
    ax.text(0.99, 0.02, "labelled: ratios above 3 or below 1", transform=ax.transAxes, fontsize=8, color=MUTED, ha="right")
    ax.legend(frameon=False)
    ax.grid(color="#E5E7EB", which="both", lw=0.4)
    fig.savefig(P("output", "fig4_team_size.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    with open(P("output", "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
