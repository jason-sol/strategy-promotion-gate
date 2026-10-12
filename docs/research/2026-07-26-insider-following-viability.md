# Insider-trading-following: viability assessment (2026-07-26)

**Verdict: NO-GO for live capital; qualified go only as a strictly-gated paper research experiment.** Deep research with independent claim verification (28 claims checked, 0 contradicted; corrections from the verification pass applied below).

## Question

Is an automated retail strategy that copies disclosed SEC Form 4 insider *purchases* (via openinsider.com, QuiverQuant, finviz, insiderscreener, or SEC EDGAR) viable and likely profitable for a ~$100k Alpaca account?

## Findings

### 1. The insider-purchase signal is real — one of the best-documented anomalies

- Lakonishok & Lee (2001, *RFS*): purchases predict positive abnormal returns, sales do not; effect driven by small firms (~7.4% abnormal in small-caps over the following year).
- Jeng, Metrick & Zeckhauser (2003, *REStat*): purchases earn >6% abnormal per year; sales insignificant.
- Buy-side only: insider *selling* is noise (liquidity/diversification/tax-driven).

### 2. Post-SOX, the post-filing window is the tradeable one — and it strengthened

Brochet (2010, *The Accounting Review*): 5-day announcement return around insider-purchase filings rose from 1.0% pre-SOX to 2.3% post-SOX (2-business-day Form 4 deadline since Aug 2002). The information a copier can act on now arrives *after* the public filing.

### 3. Filters roughly double the raw signal

- **Opportunistic vs routine** — Cohen, Malloy & Pomorski (2012, *JF*, "Decoding Inside Information"): opportunistic insiders earn value-weighted 82 bps/month (~10%/yr); routine calendar traders ≈ 0.
- **Cluster buys** — Alldredge & Blank (2019, *JFR*; 2017 working paper): purchases within 2 days of a peer insider earn 2.1% next-month abnormal vs 1.2% solitary.
- **Executive clusters** — Kang, Kim & Wang (2018): ~3.8% vs ~2.0% over 21 trading days.
- **Trend confirmation** — Zhao (2026, arXiv:2602.06198): microcap purchases after >10% prior appreciation show 6.3% mean CAR; distance from 52-week high = 36% of feature importance. *Caveat: single-author, non-peer-reviewed preprint; 6.3% is a conditional subgroup CAR, not a strategy return; does not model costs or liquidity.*
- Practical filter stack: open-market purchases only (Transaction Code P), cluster ≥2 insiders or C-suite buyer, meaningful size vs holdings, non-routine timing.

### 4. The decisive adverse finding: the edge vanishes at retail-tradeable size

Oenschläger & Möllenhoff (2025), *Finance Research Letters* 72, "Insider filings as trading signals — does it pay to be fast?" (verified near-verbatim against the source): post-filing abnormal returns **"vanish and even become negative when limiting the tradable dollar amount for each trading signal to a reasonable size"**; returns are **negatively correlated with stock liquidity**, "almost negating a potentially profitable and scalable trading strategy even before considering transaction costs." The alpha is trapped in illiquid small/microcaps where round-trip spreads run 2–5%.

### 5. Real-world reality check

The flagship insider-following ETF — Direxion All Cap Insider Sentiment (KNOW), launched Dec 2011 on a Sabrient index — was liquidated in October 2020, despite favourable back-cast index marketing. No large, durable, purely-insider-following US ETF operates today.

### 6. Legality: clear

Trading on filed Form 4s is lawful for outsiders. Rule 10b-5 liability requires material *non-public* information obtained via breach of duty; a Form 4 on EDGAR is public record, so a copier possesses no MNPI. (Caveat: this covers the public filing only — combining with privately obtained information is a different matter.)

### 7. Data access (if ever pursued)

| Source | Cost | Notes |
|---|---|---|
| SEC EDGAR (`data.sec.gov`) | Free, no key | 10 req/s global limit (verified). Form 4 XML parsed directly; most robust and ToS-clean. RSS refresh cadence (~10 min, 6am–10pm ET weekdays) is **unverified operational lore** — re-source before scheduler design. |
| QuiverQuant API | Trader tier $75/mo ($62.50/mo annual) | Insider data excluded from $30 Hobbyist tier (verified on vendor pricing). Turnkey. |
| Finnhub | Free tier, 60 calls/min | Insider transactions + MSPR sentiment; good for prototyping. |
| openinsider.com | Free (scraping) | HTML scraping — fragile, ToS-ambiguous; manual cross-check only. |
| sec-api.io | Paid | Structured Form 4 since 2009; ~11.4M transactions since 2003 — the realistic option for point-in-time backtest history. |

**The hard gate is backtest honesty**: filing-timestamp (not trade-date) point-in-time alignment, survivorship-clean prices including delisted microcaps, and real spread/slippage modeling. Free data likely cannot deliver this for the small-cap universe where the alpha lives.

### 8. Realistic expectations for a $100k account

- Gross literature alpha for filtered purchase portfolios: ~6–10%/yr — concentrated in names we can't trade cheaply.
- Defensible planning range for a liquidity-constrained version: **0 to ~3%/yr net over SPY, high variance, real probability of negative**.
- Failure modes, ranked: (1) small-cap liquidity/spread tax; (2) look-ahead/survivorship bias in backtests; (3) crowding/decay since real-time alert proliferation (~2015+); (4) regime dependence (insider buying clusters in drawdowns — partly a beta/value bet); (5) lumpy event concentration, awkward for our risk sizing.
- Our own validation harness (DSR ≥ 0.95, PBO ≤ 0.5, OOS > 0) would very likely — and correctly — reject an honest version. If it *passes*, suspect look-ahead or survivorship first.

### 9. Alternatives ranked (why we're not building this)

1. **Execution-cost reduction** on the existing book — guaranteed improvement, no signal risk. Highest EV per effort. → being built.
2. **PEAD / earnings momentum** in liquid large-caps — durable, well-documented anomaly that fits our liquidity and harness. → being built (signal-logging-only until point-in-time data exists).
3. **Congressional-trading following** — better tradability (large-caps), weaker causal edge. Live ETFs NANC/KRUZ look strong since Feb 2023 but exact cumulative figures could not be fully verified (KRUZ "+73%" unconfirmed); Belmont/Sacerdote et al. (2022, *J. Public Economics*) find members show no systematic skill; disclosure lags up to 45 days (STOCK Act).
4. **13F cloning** — 45-day delay is crippling for naive copies; filtered high-conviction cloning has shown modest, decaying alpha (exact figures in circulation are inconsistent; treat qualitatively).
5. **Insider-following** — this document.

## Gates if ever revisited (paper-only)

1. Backtest on filing timestamp with survivorship-clean, delisting-inclusive prices — non-negotiable.
2. Model per-name half-spread + slippage honestly (assume 2–5% round-trip sub-$500M).
3. Pre-register a small filter set *before* looking at returns; run through the existing DSR/CSCV/PBO harness.
4. Restrict universe to tradeable ADV/spread — expect this to remove most of the gross alpha.
5. If gates 1–2 kill the net return, stop: that is the answer.

## Sources

- Lakonishok & Lee (2001) — https://academic.oup.com/rfs/article-abstract/14/1/79/1587398
- Jeng, Metrick & Zeckhauser (2003) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=146029
- Cohen, Malloy & Pomorski (2012) — https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01740.x
- Brochet (2010) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1108731
- Alldredge & Blank (2019, JFR) — https://onlinelibrary.wiley.com/doi/abs/10.1111/jfir.12172
- Kang, Kim & Wang cluster findings — https://www.2iqresearch.com/blog/what-is-cluster-buying-and-why-is-it-such-a-powerful-insider-signal
- Oenschläger & Möllenhoff (2025), FRL 72 — https://www.sciencedirect.com/science/article/pii/S1544612324015435
- Zhao (2026) preprint — https://arxiv.org/abs/2602.06198
- KNOW ETF launch/liquidation — https://www.sec.gov/Archives/edgar/data/0001424958/000089418911005458/drxnshrs-etftrst_8a.htm ; https://www.investing.com/etfs/direxion-all-cap-ins-sent-shrs-profile
- Rule 10b-5 / MNPI basis — https://www.congress.gov/crs_external_products/LSB/HTML/LSB11119.web.html ; https://www.sec.gov/files/33-11138-fact-sheet.pdf
- EDGAR access & rate limits — https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- QuiverQuant pricing — https://api.quiverquant.com/pricing/
- sec-api.io Form 4 datasets — https://sec-api.io/datasets/form-4
- Belmont/Sacerdote et al. (2022) — https://www.sciencedirect.com/science/article/abs/pii/S0047272722000044
