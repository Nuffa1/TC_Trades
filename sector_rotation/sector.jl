module MacroSectorRotation

using DataFrames
using Statistics
using StatsBase
using Dates
using RollingFunctions

# Economic Regime Enum  Key words[MacroCycle, RegimeStates, Enum]
@enum EconomicRegime begin
    EARLY_EXPANSION
    MID_EXPANSION
    LATE_EXPANSION
    RECESSION
    RECOVERY
end

# Config Structs  Key words[Config, Weights, Settings]
struct MacroIndicatorConfig
    consumer_weight::Float64
    labor_weight::Float64
    inflation_weight::Float64
    credit_weight::Float64
end
MacroIndicatorConfig() = MacroIndicatorConfig(0.25, 0.25, 0.25, 0.25)

# Portfolio Config  Key words[Portfolio, Constraints, Rebalance]
struct PortfolioConfig
    max_weight_per_sector::Float64
    transaction_cost_bps::Float64
    rebalance_frequency::String
end
PortfolioConfig() = PortfolioConfig(0.30, 5.0, "M")

# Normalization Helper  Key words[ZScore, Normalize, Standardize]
function zscore(v)
    μ = mean(v)
    σ = std(v) + 1e-8
    return (v .- μ) ./ σ
end

# Composite Score Builder  Key words[Growth, Inflation, Stress]
function compute_composite_scores(df::DataFrame, cfg::MacroIndicatorConfig)
    consumer  = zscore(df.consumer)
    labor     = zscore(df.labor)
    inflation = zscore(df.inflation)
    credit    = zscore(df.credit)

    growth = cfg.consumer_weight .* consumer .+ cfg.labor_weight .* labor
    infl   = cfg.inflation_weight .* inflation
    stress = cfg.credit_weight .* credit

    return DataFrame(
        date=df.date,
        growth=growth,
        inflation_pressure=infl,
        financial_stress=stress
    )
end

# Regime Detection  Key words[Classifier, Thresholds, Regime]
function detect_regime_series(df::DataFrame)
    scores = compute_composite_scores(df, MacroIndicatorConfig())
    regimes = EconomicRegime[]

    for i in 1:nrow(scores)
        g = scores.growth[i]
        infl = scores.inflation_pressure[i]
        s = scores.financial_stress[i]

        reg = if g > 0.7 && s < 0
            EARLY_EXPANSION
        elseif g > 0.3 && s <= 0.3
            MID_EXPANSION
        elseif g > 0 && (infl > 0.5 || s > 0.3)
            LATE_EXPANSION
        elseif g < -0.3 && s > 0.5
            RECESSION
        else
            RECOVERY
        end

        push!(regimes, reg)
    end

    return DataFrame(date=df.date, regime=regimes)
end

# Sector Mapping  Key words[RegimeMap, Sectors, Rotation]
const REGIME_MAP = Dict(
    EARLY_EXPANSION => ["XLY","XLF","XLI","XLK"],
    MID_EXPANSION   => ["XLK","XLI","XLB"],
    LATE_EXPANSION  => ["XLE","XLF","XLP","XLU"],
    RECESSION       => ["XLP","XLU","XLV"],
    RECOVERY        => ["XLY","XLI","XLF"]
)

# Sector Getter  
get_sectors_for_regime(r::EconomicRegime) = REGIME_MAP[r]

# Momentum Calculation  Key words[Momentum, Returns, Lookback]
function compute_momentum(prices::DataFrame, lookback::Int)
    tickers = names(prices)[2:end]
    mom = Dict{String,Float64}()

    for t in tickers
        p = prices[:,t]
        if length(p) > lookback
            mom[t] = p[end]/p[end-lookback] - 1
        end
    end
    return mom
end

# Volatility Calculation 
function compute_volatility(prices::DataFrame, lookback::Int)
    tickers = names(prices)[2:end]
    vol = Dict{String,Float64}()

    for t in tickers
        p = prices[:,t]
        rets = diff(p) ./ p[1:end-1]
        vol[t] = std(last(rets, lookback))
    end

    return vol
end

# Asset Filter 
function filter_assets(prices::DataFrame; momentum_lb=126, vol_lb=63, top_n=5)
    mom = compute_momentum(prices, momentum_lb)
    vol = compute_volatility(prices, vol_lb)

    common = intersect(keys(mom), keys(vol))
    score = Dict(t => mom[t] - vol[t] for t in common)

    sorted = sort(collect(score), by=x->x[2], rev=true)
    selected = first.(sorted[1:min(top_n, length(sorted))])

    return selected
end

# Portfolio Builder 
function build_portfolio(tickers::Vector{String}, cfg::PortfolioConfig)
    if isempty(tickers)
        return Dict{String,Float64}()
    end

    n = length(tickers)
    w = min(1/n, cfg.max_weight_per_sector)

    raw = Dict(t => w for t in tickers)
    total = sum(values(raw))
    return Dict(t => raw[t]/total for t in tickers)
end

# Rebalance Dates 
function get_rebalance_dates(dates, freq="M")
    if freq == "M"
        return unique(Date(d.year, d.month, day(endofmonth(d))) for d in dates)
    end
    error("Frequency not implemented")
end

# Backtester 
    dates = prices.date
    rebal_dates = get_rebalance_dates(dates, cfg.rebalance_frequency)

    equity = zeros(Float64, length(dates))
    equity[1] = 1_000_000.0

    current_weights = Dict{String,Float64}()

    for i in 2:length(dates)
        date = dates[i]

        if date in rebal_dates
            reg_row = last(regimes[regimes.date .<= date, :], 1)
            reg = reg_row.regime[1]

            sectors = get_sectors_for_regime(reg)
            df_sel = select(prices[prices.date .<= date, :], [:date; sectors])

            allowed = filter_assets(df_sel; top_n=length(sectors))
            current_weights = build_portfolio(allowed, cfg)
        end

        if !isempty(current_weights)
            prev = prices[i-1, :]
            now  = prices[i, :]

            ret = sum(current_weights[t] * (now[t]/prev[t] - 1) for t in keys(current_weights))

            if date in rebal_dates
                ret -= cfg.transaction_cost_bps / 10_000
            end

            equity[i] = equity[i-1] * (1 + ret)
        else
            equity[i] = equity[i-1]
        end
    end

    return DataFrame(date=dates, equity=equity)
end

# Pipeline Runner  
function run_pipeline(macro_df::DataFrame, price_df::DataFrame)
    regimes = detect_regime_series(macro_df)
    cfg = PortfolioConfig()
    eq = backtest(price_df, regimes, cfg)
    return eq, regimes
end

end # module
