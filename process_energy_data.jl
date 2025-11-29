"""
Global Energy Consumption Audit - Julia Script
BP Statistical Review Dataset Processing Pipeline

This script processes raw energy consumption data, standardizes units,
cleans missing values, normalizes by population/GDP, and generates
summary statistics and visualizations.
"""

using DataFrames
using CSV
using Statistics
using Printf
using Dates
using Plots
using StatsPlots
gr()  # Use GR backend

# Unit conversion constants
const EJ_TO_TWH = 277.778
const MTOE_TO_TWH = 11.63
const KTOE_TO_TWH = 0.01163

# Sanity check thresholds
const MAX_TWH = 50000.0  # Maximum reasonable energy consumption in TWh
const MIN_TWH = 0.0      # Minimum (negative values are invalid)

# Data quality flags
const FLAG_ORIGINAL = 0
const FLAG_INTERPOLATED = 1
const FLAG_MISSING_BLOCK = 2
const FLAG_REMOVED_OUTLIER = 3


"""
    load_data(file_path::String) -> DataFrame

Load raw energy data from CSV or Excel file.
"""
function load_data(file_path::String)::DataFrame
    println("Loading data from $file_path")
    
    if endswith(file_path, ".csv")
        df = CSV.read(file_path, DataFrame)
    elseif endswith(file_path, ".xlsx") || endswith(file_path, ".xls")
        error("Excel files not directly supported. Please convert to CSV or use XLSX.jl")
    else
        error("Unsupported file format: $file_path")
    end
    
    println("Loaded $(nrow(df)) rows, $(ncol(df)) columns")
    return df
end


"""
    standardize_columns(df::DataFrame) -> DataFrame

Standardize column names to lowercase snake_case.
"""
function standardize_columns(df::DataFrame)::DataFrame
    println("Standardizing column names")
    
    df = copy(df)
    
    # Convert column names to lowercase and replace spaces/special chars
    new_names = String[]
    for name in names(df)
        # First convert to lowercase, then replace spaces/dashes, then remove non-alphanumeric
        new_name = lowercase(string(name))
        new_name = replace(new_name, ' ' => '_', '-' => '_')
        new_name = replace(new_name, r"[^a-z0-9_]" => "")
        push!(new_names, new_name)
    end
    
    rename!(df, Symbol.(names(df)) .=> Symbol.(new_names))
    
    # Common column name mappings
    column_mapping = Dict(
        "country" => "country",
        "year" => "year",
        "primary_energy" => "primary_energy",
        "primary_energy_consumption" => "primary_energy",
        "energy_consumption" => "primary_energy",
        "electricity_generation" => "electricity_generation",
        "electricity" => "electricity_generation",
        "total_energy" => "primary_energy",
    )
    
    for (old_name, new_name) in column_mapping
        if hasproperty(df, Symbol(old_name)) && !hasproperty(df, Symbol(new_name))
            rename!(df, Symbol(old_name) => Symbol(new_name))
        end
    end
    
    println("Standardized columns: $(names(df))")
    return df
end


"""
    convert_units_to_twh(df::DataFrame) -> DataFrame

Convert all energy units to TWh.
"""
function convert_units_to_twh(df::DataFrame)::DataFrame
    println("Converting units to TWh")
    
    df = copy(df)
    
    # Identify energy columns (excluding country, year, and flag columns)
    exclude_cols = ["country", "year", "data_quality_flag", "region"]
    energy_cols = [col for col in names(df) 
                   if col ∉ exclude_cols && 
                      eltype(df[!, col]) <: Union{Real, Missing}]
    
    # Detect unit from column name or data magnitude
    for col in energy_cols
        col_lower = lowercase(string(col))
        
        if occursin("ej", col_lower) || occursin("exajoule", col_lower)
            # Convert EJ to TWh
            if any(!ismissing, df[!, col])
                df[!, col] = df[!, col] .* EJ_TO_TWH
                println("Converted $col from EJ to TWh")
            end
        
        elseif occursin("mtoe", col_lower) || occursin("million_tonnes", col_lower)
            # Convert Mtoe to TWh
            if any(!ismissing, df[!, col])
                df[!, col] = df[!, col] .* MTOE_TO_TWH
                println("Converted $col from Mtoe to TWh")
            end
        
        elseif occursin("ktoe", col_lower) || occursin("thousand_tonnes", col_lower)
            # Convert ktoe to TWh
            if any(!ismissing, df[!, col])
                df[!, col] = df[!, col] .* KTOE_TO_TWH
                println("Converted $col from ktoe to TWh")
            end
        
        elseif !occursin("twh", col_lower) && any(!ismissing, df[!, col])
            # Try to detect unit by magnitude
            max_val = maximum(skipmissing(df[!, col]))
            if max_val > 1000 && max_val < 10000
                # Likely EJ
                df[!, col] = df[!, col] .* EJ_TO_TWH
                println("Auto-converted $col from EJ to TWh (detected by magnitude)")
            elseif max_val > 10 && max_val < 1000
                # Likely Mtoe
                df[!, col] = df[!, col] .* MTOE_TO_TWH
                println("Auto-converted $col from Mtoe to TWh (detected by magnitude)")
            end
        end
    end
    
    # Create unified total_energy_twh column
    # Priority: primary_energy > electricity_generation > first available energy column
    # Avoid double-counting when multiple unit columns exist for same data
    if hasproperty(df, :primary_energy)
        df[!, :total_energy_twh] = df[!, :primary_energy]
    elseif hasproperty(df, :electricity_generation)
        df[!, :total_energy_twh] = df[!, :electricity_generation]
    else
        # If multiple energy columns exist, use first non-null value per row to avoid double-counting
        numeric_cols = [col for col in energy_cols if col ∉ exclude_cols]
        if length(numeric_cols) == 1
            df[!, :total_energy_twh] = df[!, Symbol(numeric_cols[1])]
        elseif length(numeric_cols) > 1
            # Use first non-null value across columns for each row
            println("Multiple energy columns found. Using first non-null value per row to avoid double-counting")
            df[!, :total_energy_twh] = [
                begin
                    val = missing
                    for col in numeric_cols
                        col_val = df[row, Symbol(col)]
                        if !ismissing(col_val)
                            val = col_val
                            break
                        end
                    end
                    val
                end
                for row in 1:nrow(df)
            ]
        else
            error("No energy columns found in the dataset")
        end
    end
    
    # Sanity check: verify conversion
    if any(!ismissing, df[!, :total_energy_twh])
        max_val = maximum(skipmissing(df[!, :total_energy_twh]))
        min_val = minimum(skipmissing(df[!, :total_energy_twh]))
        println(@sprintf("Total energy range: %.2f - %.2f TWh", min_val, max_val))
        
        # Assert reasonable values (relaxed threshold for global totals)
        # Global energy consumption can exceed 100,000 TWh when summing all countries
        if max_val > MAX_TWH * 5  # 250,000 TWh threshold
            println("WARNING: Very high maximum value $max_val TWh detected - may indicate double-counting or data issue")
        end
        @assert min_val >= MIN_TWH "Minimum value $min_val TWh is negative"
    end
    
    return df
end


"""
    clean_data(df::DataFrame) -> Tuple{DataFrame, DataFrame}

Clean data: handle missing values, interpolate, flag issues, remove outliers.
"""
function clean_data(df::DataFrame)::Tuple{DataFrame, DataFrame}
    println("Cleaning data")
    
    df = copy(df)
    
    # Initialize data quality flag column
    if !hasproperty(df, :data_quality_flag)
        df[!, :data_quality_flag] = fill(FLAG_ORIGINAL, nrow(df))
    end
    
    # Identify missing values by country
    missing_by_country = combine(
        groupby(df, :country),
        :total_energy_twh => (x -> sum(ismissing.(x))) => :missing_count
    )
    sort!(missing_by_country, :missing_count, rev=true)
    println("Missing values by country:")
    println(missing_by_country[missing_by_country.missing_count .> 0, :])
    
    # Track outliers
    outlier_rows = DataFrame(
        country = String[],
        year = Int[],
        value = Float64[],
        reason = String[]
    )
    
    # Process each country
    for country in unique(df[!, :country])
        country_mask = df[!, :country] .== country
        country_indices = findall(country_mask)
        country_data = df[country_mask, :]
        
        # Check for impossible values (handle missing values)
        energy_values = df[!, :total_energy_twh]
        invalid_mask = fill(false, nrow(df))
        for i in 1:nrow(df)
            if country_mask[i] && !ismissing(energy_values[i])
                invalid_mask[i] = (energy_values[i] < MIN_TWH) || (energy_values[i] > MAX_TWH)
            end
        end
        
        if any(invalid_mask)
            invalid_indices = findall(invalid_mask)
            println("Found $(sum(invalid_mask)) outliers in $country")
            
            for idx in invalid_indices
                push!(outlier_rows, (
                    country,
                    df[idx, :year],
                    df[idx, :total_energy_twh],
                    "outlier"
                ))
            end
            
            # Mark as removed
            df[invalid_indices, :data_quality_flag] .= FLAG_REMOVED_OUTLIER
            df[invalid_indices, :total_energy_twh] .= missing
        end
        
        # Check for missing data patterns
        energy_series = country_data[!, :total_energy_twh]
        missing_count = sum(ismissing.(energy_series))
        total_count = length(energy_series)
        
        if missing_count == total_count
            # Entire country block missing
            println("Entire country block missing for $country")
            df[country_indices, :data_quality_flag] .= FLAG_MISSING_BLOCK
        
        elseif missing_count > 0 && missing_count < total_count
            # Partial missing years - interpolate
            println("Interpolating $missing_count missing values for $country")
            
            # Sort by year for interpolation
            sorted_indices = sortperm(country_data[!, :year])
            sorted_country_indices = country_indices[sorted_indices]
            
            # Interpolate
            energy_values = df[sorted_country_indices, :total_energy_twh]
            
            # Simple linear interpolation
            for i in 2:(length(energy_values)-1)
                if ismissing(energy_values[i])
                    # Find nearest non-missing values
                    prev_val = missing
                    next_val = missing
                    prev_idx = i - 1
                    next_idx = i + 1
                    
                    while prev_idx >= 1 && ismissing(energy_values[prev_idx])
                        prev_idx -= 1
                    end
                    if prev_idx >= 1
                        prev_val = energy_values[prev_idx]
                    end
                    
                    while next_idx <= length(energy_values) && ismissing(energy_values[next_idx])
                        next_idx += 1
                    end
                    if next_idx <= length(energy_values)
                        next_val = energy_values[next_idx]
                    end
                    
                    if !ismissing(prev_val) && !ismissing(next_val)
                        # Linear interpolation
                        year_prev = df[sorted_country_indices[prev_idx], :year]
                        year_next = df[sorted_country_indices[next_idx], :year]
                        year_curr = df[sorted_country_indices[i], :year]
                        
                        if year_next != year_prev
                            weight = (year_curr - year_prev) / (year_next - year_prev)
                            interpolated_val = prev_val + weight * (next_val - prev_val)
                            df[sorted_country_indices[i], :total_energy_twh] = interpolated_val
                            df[sorted_country_indices[i], :data_quality_flag] = FLAG_INTERPOLATED
                        end
                    end
                end
            end
        end
    end
    
    # Remove rows marked as outliers
    df_cleaned = df[df[!, :data_quality_flag] .!= FLAG_REMOVED_OUTLIER, :]
    
    println("Cleaned data: $(nrow(df_cleaned)) rows remaining ($(nrow(df) - nrow(df_cleaned)) removed)")
    
    return df_cleaned, outlier_rows
end


"""
    load_population_data(file_path::Union{String, Nothing}) -> DataFrame

Load population data. If file not provided, create sample.
"""
function load_population_data(file_path::Union{String, Nothing} = nothing)::DataFrame
    if file_path !== nothing && isfile(file_path)
        println("Loading population data from $file_path")
        pop_df = CSV.read(file_path, DataFrame)
        rename!(pop_df, [lowercase(replace(name, ' ' => '_')) for name in names(pop_df)])
        return pop_df
    end
    
    # Create sample population data (in millions)
    println("No population file provided, creating sample data")
    countries = ["United States", "China", "India", "Japan", "Germany", 
                 "Russia", "Brazil", "South Korea", "Canada", "France"]
    years = 2000:2024
    
    base_pop = Dict(
        "United States" => 280.0,
        "China" => 1400.0,
        "India" => 1200.0,
        "Japan" => 125.0,
        "Germany" => 83.0,
        "Russia" => 145.0,
        "Brazil" => 200.0,
        "South Korea" => 52.0,
        "Canada" => 38.0,
        "France" => 67.0
    )
    
    pop_data = DataFrame(
        country = String[],
        year = Int[],
        population = Float64[]
    )
    
    for country in countries
        for year in years
            growth_rate = 0.01  # 1% annual growth
            years_since_2000 = year - 2000
            pop = base_pop[country] * (1 + growth_rate) ^ years_since_2000
            push!(pop_data, (country, year, pop))
        end
    end
    
    return pop_data
end


"""
    normalize_data(df::DataFrame, pop_df::Union{DataFrame, Nothing} = nothing,
                   gdp_df::Union{DataFrame, Nothing} = nothing) -> DataFrame

Normalize energy data by population and GDP.
"""
function normalize_data(df::DataFrame, 
                       pop_df::Union{DataFrame, Nothing} = nothing,
                       gdp_df::Union{DataFrame, Nothing} = nothing)::DataFrame
    println("Normalizing data by population and GDP")
    
    df = copy(df)
    
    # Normalize by population
    if pop_df !== nothing
        # Merge population data
        df = leftjoin(df, pop_df[!, [:country, :year, :population]], on = [:country, :year])
        
        # Calculate per-capita energy
        df[!, :energy_per_capita_twh] = df[!, :total_energy_twh] ./ df[!, :population]
        # Replace Inf and -Inf with missing
        df[!, :energy_per_capita_twh] = [isfinite(x) ? x : missing 
                                         for x in df[!, :energy_per_capita_twh]]
        
        println("Added energy_per_capita_twh column")
    else
        pop_df = load_population_data()
        return normalize_data(df, pop_df, gdp_df)
    end
    
    # Normalize by GDP if available
    if gdp_df !== nothing
        rename!(gdp_df, [lowercase(replace(name, ' ' => '_')) for name in names(gdp_df)])
        df = leftjoin(df, gdp_df[!, [:country, :year, :gdp]], on = [:country, :year])
        
        # Calculate energy intensity
        df[!, :energy_intensity_twh_per_gdp] = df[!, :total_energy_twh] ./ df[!, :gdp]
        # Replace Inf and -Inf with missing
        df[!, :energy_intensity_twh_per_gdp] = [isfinite(x) ? x : missing 
                                                 for x in df[!, :energy_intensity_twh_per_gdp]]
        
        println("Added energy_intensity_twh_per_gdp column")
    end
    
    return df
end


"""
    generate_summary_statistics(df::DataFrame) -> Dict

Generate summary statistics.
"""
function generate_summary_statistics(df::DataFrame)::Dict
    println("Generating summary statistics")
    
    stats = Dict()
    
    # Get latest year
    latest_year = maximum(df[!, :year])
    stats["latest_year"] = latest_year
    
    # Top 10 energy consumers (latest year)
    latest_data = df[df[!, :year] .== latest_year, :]
    sort!(latest_data, :total_energy_twh, rev=true)
    top_10 = latest_data[1:min(10, nrow(latest_data)), [:country, :total_energy_twh]]
    if hasproperty(latest_data, :energy_per_capita_twh)
        top_10[!, :energy_per_capita_twh] = latest_data[1:min(10, nrow(latest_data)), :energy_per_capita_twh]
    end
    stats["top_10_consumers"] = top_10
    
    # Growth rates 2000-2024 by country
    if 2000 in df[!, :year] && latest_year >= 2000
        growth_data = DataFrame(
            country = String[],
            growth_rate_pct = Float64[],
            value_2000 = Float64[],
            value_latest = Float64[]
        )
        
        for country in unique(df[!, :country])
            country_data = df[df[!, :country] .== country, :]
            sort!(country_data, :year)
            
            year_2000_data = country_data[country_data[!, :year] .== 2000, :]
            year_latest_data = country_data[country_data[!, :year] .== latest_year, :]
            
            if nrow(year_2000_data) > 0 && nrow(year_latest_data) > 0
                val_2000 = year_2000_data[1, :total_energy_twh]
                val_latest = year_latest_data[1, :total_energy_twh]
                
                if !ismissing(val_2000) && !ismissing(val_latest) && val_2000 > 0
                    growth_rate = ((val_latest / val_2000) ^ (1 / (latest_year - 2000)) - 1) * 100
                    push!(growth_data, (country, growth_rate, val_2000, val_latest))
                end
            end
        end
        
        sort!(growth_data, :growth_rate_pct, rev=true)
        stats["growth_rates"] = growth_data
    end
    
    # Missing data report
    missing_report = combine(
        groupby(df, :country),
        :total_energy_twh => (x -> sum(ismissing.(x))) => :missing_values,
        :data_quality_flag => (x -> sum(x .== FLAG_MISSING_BLOCK)) => :missing_blocks
    )
    filter!(row -> row.missing_values > 0, missing_report)
    sort!(missing_report, :missing_values, rev=true)
    stats["missing_data_report"] = missing_report
    
    # Data quality summary
    quality_summary = combine(groupby(df, :data_quality_flag), nrow => :count)
    quality_labels = Dict(FLAG_ORIGINAL => "original", 
                         FLAG_INTERPOLATED => "interpolated",
                         FLAG_MISSING_BLOCK => "flagged_missing_block",
                         FLAG_REMOVED_OUTLIER => "removed_outlier")
    stats["data_quality_summary"] = quality_summary
    
    return stats
end


"""
    create_visualizations(df::DataFrame, output_dir::String = "figures")

Create visualizations: time series, scatter plots, heatmap, bar charts.
"""
function create_visualizations(df::DataFrame, output_dir::String = "figures")
    println("Creating visualizations in $output_dir")
    
    mkpath(output_dir)
    
    # 1. Time series for selected countries
    latest_data = df[df[!, :year] .== maximum(df[!, :year]), :]
    sort!(latest_data, :total_energy_twh, rev=true)
    selected_countries = latest_data[1:min(5, nrow(latest_data)), :country]
    
    p1 = plot(legend=:topleft, size=(800, 400))
    for country in selected_countries
        country_data = df[df[!, :country] .== country, :]
        sort!(country_data, :year)
        plot!(p1, country_data[!, :year], country_data[!, :total_energy_twh], 
              label=country, marker=:circle, linewidth=2)
    end
    xlabel!(p1, "Year")
    ylabel!(p1, "Total Energy Consumption (TWh)")
    title!(p1, "Energy Consumption Time Series - Top 5 Countries")
    savefig(p1, "$output_dir/time_series_top_countries.png")
    
    # 2. Scatter: GDP vs Energy Use (log scale) - if GDP available
    if hasproperty(df, :gdp)
        latest_data = df[df[!, :year] .== maximum(df[!, :year]), :]
        latest_data = latest_data[.!(ismissing.(latest_data[!, :total_energy_twh]) .| 
                                     ismissing.(latest_data[!, :gdp])), :]
        
        p2 = scatter(latest_data[!, :gdp], latest_data[!, :total_energy_twh],
                    xscale=:log10, yscale=:log10, alpha=0.6, size=(600, 500))
        xlabel!(p2, "GDP (log scale)")
        ylabel!(p2, "Energy Consumption (TWh, log scale)")
        title!(p2, "GDP vs Energy Consumption (Latest Year)")
        savefig(p2, "$output_dir/gdp_vs_energy_scatter.png")
    end
    
    # 3. Bar chart: Energy per capita
    if hasproperty(df, :energy_per_capita_twh)
        latest_data = df[df[!, :year] .== maximum(df[!, :year]), :]
        latest_data = latest_data[.!ismissing.(latest_data[!, :energy_per_capita_twh]), :]
        sort!(latest_data, :energy_per_capita_twh, rev=true)
        top_20 = latest_data[1:min(20, nrow(latest_data)), :]
        
        p3 = bar(top_20[!, :country], top_20[!, :energy_per_capita_twh],
                orientation=:horizontal, size=(800, 600), color=:steelblue)
        xlabel!(p3, "Energy per Capita (TWh)")
        title!(p3, "Top 20 Countries by Energy Consumption per Capita")
        savefig(p3, "$output_dir/energy_per_capita_bar.png")
    end
    
    # 4. Bar chart: Energy consumption by region or top countries
    if hasproperty(df, :region)
        latest_data = df[df[!, :year] .== maximum(df[!, :year]), :]
        regional_totals = combine(
            groupby(latest_data, :region),
            :total_energy_twh => sum => :total_energy_twh
        )
        sort!(regional_totals, :total_energy_twh, rev=true)
        
        p4 = bar(regional_totals[!, :region], regional_totals[!, :total_energy_twh],
                size=(600, 400), color=:coral, xrotation=45)
        ylabel!(p4, "Total Energy Consumption (TWh)")
        title!(p4, "Energy Consumption by Region ($(maximum(df[!, :year])))")
        savefig(p4, "$output_dir/energy_by_region_bar.png")
    else
        latest_data = df[df[!, :year] .== maximum(df[!, :year]), :]
        sort!(latest_data, :total_energy_twh, rev=true)
        top_countries = latest_data[1:min(15, nrow(latest_data)), :]
        
        p4 = bar(top_countries[!, :country], top_countries[!, :total_energy_twh],
                orientation=:horizontal, size=(800, 500), color=:teal)
        xlabel!(p4, "Total Energy Consumption (TWh)")
        title!(p4, "Top 15 Countries by Energy Consumption ($(maximum(df[!, :year])))")
        savefig(p4, "$output_dir/top_countries_bar.png")
    end
    
    println("Visualizations saved to $output_dir/")
end


"""
    export_results(df::DataFrame, stats::Dict, outlier_report::DataFrame,
                   output_dir::String = "output")

Export cleaned data, summary statistics, and reports.
"""
function export_results(df::DataFrame, stats::Dict, outlier_report::DataFrame,
                       output_dir::String = "output")
    println("Exporting results to $output_dir")
    
    mkpath(output_dir)
    
    # Export cleaned dataset
    CSV.write("$output_dir/cleaned_energy_data.csv", df)
    println("Exported cleaned data to $output_dir/cleaned_energy_data.csv")
    
    # Export summary statistics
    summary_lines = String[]
    push!(summary_lines, "=" ^ 80)
    push!(summary_lines, "GLOBAL ENERGY CONSUMPTION AUDIT - SUMMARY STATISTICS")
    push!(summary_lines, "=" ^ 80)
    push!(summary_lines, "\nLatest Year: $(stats["latest_year"])")
    
    push!(summary_lines, "\n" * "-" ^ 80)
    push!(summary_lines, "TOP 10 ENERGY CONSUMERS (Latest Year)")
    push!(summary_lines, "-" ^ 80)
    push!(summary_lines, string(stats["top_10_consumers"]))
    
    if haskey(stats, "growth_rates") && nrow(stats["growth_rates"]) > 0
        push!(summary_lines, "\n" * "-" ^ 80)
        push!(summary_lines, "GROWTH RATES (2000-2024)")
        push!(summary_lines, "-" ^ 80)
        push!(summary_lines, string(first(stats["growth_rates"], 20)))
    end
    
    push!(summary_lines, "\n" * "-" ^ 80)
    push!(summary_lines, "DATA QUALITY SUMMARY")
    push!(summary_lines, "-" ^ 80)
    push!(summary_lines, string(stats["data_quality_summary"]))
    
    if nrow(stats["missing_data_report"]) > 0
        push!(summary_lines, "\n" * "-" ^ 80)
        push!(summary_lines, "MISSING DATA REPORT")
        push!(summary_lines, "-" ^ 80)
        push!(summary_lines, string(stats["missing_data_report"]))
    end
    
    # Write summary to file
    open("$output_dir/summary_statistics.txt", "w") do f
        write(f, join(summary_lines, "\n"))
    end
    
    # Export CSV summaries
    CSV.write("$output_dir/top_10_consumers.csv", stats["top_10_consumers"])
    
    if haskey(stats, "growth_rates") && nrow(stats["growth_rates"]) > 0
        CSV.write("$output_dir/growth_rates.csv", stats["growth_rates"])
    end
    
    if nrow(stats["missing_data_report"]) > 0
        CSV.write("$output_dir/missing_data_report.csv", stats["missing_data_report"])
    end
    
    if nrow(outlier_report) > 0
        CSV.write("$output_dir/outlier_report.csv", outlier_report)
    end
    
    println("All results exported to $output_dir/")
end


"""
    main()

Main execution function.
"""
function main()
    # Configuration
    INPUT_FILE = "raw_energy_data.csv"  # Change this to your input file
    POPULATION_FILE = nothing  # Optional: path to population CSV
    GDP_FILE = nothing  # Optional: path to GDP CSV
    
    # Check if input file exists
    if !isfile(INPUT_FILE)
        println("WARNING: Input file $INPUT_FILE not found. Please provide your data file.")
        println("Expected columns: country, year, primary_energy (or similar), units in EJ/Mtoe/ktoe/TWh")
        return
    end
    
    # Load and process data
    df_raw = load_data(INPUT_FILE)
    df_std = standardize_columns(df_raw)
    df_converted = convert_units_to_twh(df_std)
    df_cleaned, outlier_report = clean_data(df_converted)
    
    # Load population data
    pop_df = load_population_data(POPULATION_FILE)
    
    # Load GDP data if available
    gdp_df = nothing
    if GDP_FILE !== nothing && isfile(GDP_FILE)
        gdp_df = CSV.read(GDP_FILE, DataFrame)
    end
    
    # Normalize data
    df_normalized = normalize_data(df_cleaned, pop_df, gdp_df)
    
    # Generate summary statistics
    stats = generate_summary_statistics(df_normalized)
    
    # Create visualizations
    create_visualizations(df_normalized)
    
    # Export results
    export_results(df_normalized, stats, outlier_report)
    
    println("Processing complete!")
end


# Run main if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

