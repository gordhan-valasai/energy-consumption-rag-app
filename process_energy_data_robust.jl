"""
Global Energy Consumption Audit - ROBUST Julia Script
BP Statistical Review Dataset Processing Pipeline

RESEARCH-GRADE VERSION with:
- Explicit unit handling (no magnitude-based auto-detection)
- Schema validation
- Robust outlier detection (MAD/IQR/Z-score)
- Gap-limited interpolation
- Comprehensive audit logging
- No synthetic data generation
- Configuration file support
"""

using DataFrames
using CSV
using Statistics
using Printf
using Dates
using YAML
using JSON

# Data quality flags
struct DataQualityFlags
    ORIGINAL::Int
    INTERPOLATED::Int
    FLAGGED_MISSING_BLOCK::Int
    REMOVED_OUTLIER::Int
    INVALID_UNIT::Int
    SCHEMA_VIOLATION::Int
end

const FLAGS = DataQualityFlags(0, 1, 2, 3, 4, 5)

# Logging helper
function log_info(msg::String)
    println("[INFO] $(now()) | $msg")
end

function log_warn(msg::String)
    println("[WARN] $(now()) | $msg")
end

function log_error(msg::String)
    println("[ERROR] $(now()) | $msg")
end

function log_debug(msg::String)
    println("[DEBUG] $(now()) | $msg")
end


"""
    SchemaValidator

Validates data schema and structure.
"""
mutable struct SchemaValidator
    config::Dict
    errors::Vector{String}
    warnings::Vector{String}
end

function SchemaValidator(config::Dict)
    schema_config = get(config, "schema", Dict())
    return SchemaValidator(config, String[], String[])
end

function validate(validator::SchemaValidator, df::DataFrame)::Tuple{Bool, Vector{String}, Vector{String}}
    validator.errors = String[]
    validator.warnings = String[]
    
    schema_config = get(validator.config, "schema", Dict())
    
    # Check required columns
    required = get(schema_config, "required_columns", ["country", "year"])
    missing_cols = [col for col in required if !hasproperty(df, Symbol(col))]
    if !isempty(missing_cols)
        push!(validator.errors, "Missing required columns: $(missing_cols)")
    end
    
    # Check data types
    if haskey(schema_config, "column_types")
        for (col, expected_type) in schema_config["column_types"]
            if hasproperty(df, Symbol(col))
                col_data = df[!, Symbol(col)]
                if expected_type == "integer" && !(eltype(col_data) <: Integer)
                    push!(validator.warnings, "Column $col expected integer type")
                elseif expected_type == "float" && !(eltype(col_data) <: Union{Real, Missing})
                    push!(validator.warnings, "Column $col expected float type")
                elseif expected_type == "string" && !(eltype(col_data) <: Union{String, Missing})
                    push!(validator.warnings, "Column $col expected string type")
                end
            end
        end
    end
    
    # Check year range
    if hasproperty(df, :year) && haskey(schema_config, "year_range")
        year_range = schema_config["year_range"]
        min_year = get(year_range, "min", 1900)
        max_year = get(year_range, "max", 2100)
        invalid_years = filter(row -> row.year < min_year || row.year > max_year, df)
        if nrow(invalid_years) > 0
            push!(validator.warnings, "Found $(nrow(invalid_years)) rows with years outside [$min_year, $max_year]")
        end
    end
    
    # Check for duplicate country-year pairs
    if hasproperty(df, :country) && hasproperty(df, :year)
        duplicates = nonunique(df, [:country, :year])
        if any(duplicates)
            push!(validator.errors, "Found $(sum(duplicates)) duplicate country-year pairs")
        end
    end
    
    # Check for missing values in required columns
    for col in required
        if hasproperty(df, Symbol(col))
            missing_count = sum(ismissing.(df[!, Symbol(col)]))
            if missing_count > 0
                push!(validator.errors, "Column $col has $missing_count missing values")
            end
        end
    end
    
    return isempty(validator.errors), validator.errors, validator.warnings
end


"""
    UnitConverter

Handles unit conversion with explicit unit specification.
"""
struct UnitConverter
    config::Dict
    conversions::Dict{String, Float64}
    unit_map::Dict{String, String}
end

function UnitConverter(config::Dict)
    conversions = get(config, "unit_conversions", Dict(
        "EJ_to_TWh" => 277.778,
        "Mtoe_to_TWh" => 11.63,
        "ktoe_to_TWh" => 0.01163,
        "TWh_to_TWh" => 1.0
    ))
    
    unit_map = Dict(
        "ej" => "EJ_to_TWh",
        "exajoule" => "EJ_to_TWh",
        "mtoe" => "Mtoe_to_TWh",
        "million_tonnes" => "Mtoe_to_TWh",
        "ktoe" => "ktoe_to_TWh",
        "thousand_tonnes" => "ktoe_to_TWh",
        "twh" => "TWh_to_TWh",
        "terawatt_hour" => "TWh_to_TWh"
    )
    
    return UnitConverter(config, conversions, unit_map)
end

function detect_unit_from_column_name(converter::UnitConverter, col_name::String)::Union{String, Nothing}
    """Detect unit from column name (explicit only, no magnitude guessing)."""
    col_lower = lowercase(col_name)
    for (unit_key, conversion_key) in converter.unit_map
        if occursin(unit_key, col_lower)
            return conversion_key
        end
    end
    return nothing
end

function convert_column(converter::UnitConverter, df::DataFrame, col::Symbol, unit::Union{String, Nothing} = nothing)::Vector{Union{Float64, Missing}}
    """Convert a column to TWh."""
    if unit === nothing
        unit = detect_unit_from_column_name(converter, string(col))
    end
    
    if unit === nothing
        log_warn("Could not determine unit for column $col. Assuming TWh.")
        return [ismissing(x) ? missing : Float64(x) for x in df[!, col]]
    end
    
    if !haskey(converter.conversions, unit)
        log_error("Unknown unit conversion: $unit")
        throw(ArgumentError("Unknown unit conversion: $unit"))
    end
    
    conversion_factor = converter.conversions[unit]
    converted = [ismissing(x) ? missing : Float64(x) * conversion_factor for x in df[!, col]]
    
    log_info("Converted $col from $unit to TWh (factor: $conversion_factor)")
    return converted
end

function create_unified_column(converter::UnitConverter, df::DataFrame, energy_columns::Vector{Symbol})::Vector{Union{Float64, Missing}}
    """Create unified energy column from multiple columns."""
    if isempty(energy_columns)
        throw(ArgumentError("No energy columns found"))
    end
    
    if length(energy_columns) == 1
        return convert_column(converter, df, energy_columns[1])
    end
    
    # Convert all columns to TWh first
    converted_cols = Dict{Symbol, Vector{Union{Float64, Missing}}}()
    for col in energy_columns
        converted_cols[col] = convert_column(converter, df, col)
    end
    
    # Use first non-null value per row
    result = Vector{Union{Float64, Missing}}(undef, nrow(df))
    for i in 1:nrow(df)
        val = missing
        for col in energy_columns
            col_val = converted_cols[col][i]
            if !ismissing(col_val)
                val = col_val
                break
            end
        end
        result[i] = val
    end
    
    log_info("Created unified column from $(length(energy_columns)) energy columns")
    return result
end


"""
    RobustOutlierDetector

Robust outlier detection using MAD, IQR, or Z-score.
"""
struct RobustOutlierDetector
    method::String
    mad_multiplier::Float64
    iqr_multiplier::Float64
    zscore_threshold::Float64
    fixed_threshold::Float64
end

function RobustOutlierDetector(config::Dict)
    dq_config = get(config, "data_quality", Dict())
    method = get(dq_config, "outlier_method", "mad")
    mad_multiplier = get(dq_config, "mad_multiplier", 3.0)
    iqr_multiplier = get(dq_config, "iqr_multiplier", 1.5)
    zscore_threshold = get(dq_config, "zscore_threshold", 3.0)
    fixed_threshold = get(dq_config, "max_energy_twh", 50000.0)
    
    return RobustOutlierDetector(method, mad_multiplier, iqr_multiplier, zscore_threshold, fixed_threshold)
end

function detect_mad(detector::RobustOutlierDetector, series::Vector{Union{Float64, Missing}})::Vector{Bool}
    """Median Absolute Deviation method."""
    valid_values = [x for x in series if !ismissing(x)]
    if isempty(valid_values)
        return fill(false, length(series))
    end
    
    median_val = median(valid_values)
    mad = median([abs(x - median_val) for x in valid_values])
    
    if mad == 0
        return fill(false, length(series))
    end
    
    threshold = median_val + detector.mad_multiplier * mad
    return [!ismissing(x) && x > threshold for x in series]
end

function detect_iqr(detector::RobustOutlierDetector, series::Vector{Union{Float64, Missing}})::Vector{Bool}
    """Interquartile Range method."""
    valid_values = [x for x in series if !ismissing(x)]
    if isempty(valid_values)
        return fill(false, length(series))
    end
    
    Q1 = quantile(valid_values, 0.25)
    Q3 = quantile(valid_values, 0.75)
    IQR = Q3 - Q1
    
    if IQR == 0
        return fill(false, length(series))
    end
    
    lower_bound = Q1 - detector.iqr_multiplier * IQR
    upper_bound = Q3 + detector.iqr_multiplier * IQR
    
    return [!ismissing(x) && (x < lower_bound || x > upper_bound) for x in series]
end

function detect_zscore(detector::RobustOutlierDetector, series::Vector{Union{Float64, Missing}})::Vector{Bool}
    """Z-score method."""
    valid_values = [x for x in series if !ismissing(x)]
    if isempty(valid_values)
        return fill(false, length(series))
    end
    
    mean_val = mean(valid_values)
    std_val = std(valid_values)
    
    if std_val == 0
        return fill(false, length(series))
    end
    
    z_scores = [abs((x - mean_val) / std_val) for x in series]
    return [!ismissing(x) && z_scores[i] > detector.zscore_threshold for (i, x) in enumerate(series)]
end

function detect_fixed(detector::RobustOutlierDetector, series::Vector{Union{Float64, Missing}})::Vector{Bool}
    """Fixed threshold method."""
    return [!ismissing(x) && (x < 0 || x > detector.fixed_threshold) for x in series]
end

function detect(detector::RobustOutlierDetector, series::Vector{Union{Float64, Missing}}, 
                group_by::Union{Vector{String}, Nothing} = nothing)::Vector{Bool}
    """Detect outliers in series."""
    if group_by === nothing
        # Detect outliers globally
        if detector.method == "mad"
            return detect_mad(detector, series)
        elseif detector.method == "iqr"
            return detect_iqr(detector, series)
        elseif detector.method == "zscore"
            return detect_zscore(detector, series)
        else
            return detect_fixed(detector, series)
        end
    else
        # Detect outliers per group
        outlier_mask = fill(false, length(series))
        unique_groups = unique(group_by)
        
        for group_val in unique_groups
            group_indices = findall(x -> x == group_val, group_by)
            group_series = [series[i] for i in group_indices]
            
            if detector.method == "mad"
                group_outliers = detect_mad(detector, group_series)
            elseif detector.method == "iqr"
                group_outliers = detect_iqr(detector, group_series)
            elseif detector.method == "zscore"
                group_outliers = detect_zscore(detector, group_series)
            else
                group_outliers = detect_fixed(detector, group_series)
            end
            
            for (idx, is_outlier) in zip(group_indices, group_outliers)
                outlier_mask[idx] = is_outlier
            end
        end
        
        return outlier_mask
    end
end


"""
    SafeInterpolator

Safe interpolation with gap limits.
"""
struct SafeInterpolator
    max_gap::Int
    min_years::Int
end

function SafeInterpolator(config::Dict)
    dq_config = get(config, "data_quality", Dict())
    max_gap = get(dq_config, "max_interpolation_gap", 3)
    min_years = get(dq_config, "min_years_for_interpolation", 2)
    return SafeInterpolator(max_gap, min_years)
end

function interpolate(interpolator::SafeInterpolator, df::DataFrame, value_col::Symbol,
                    group_col::Symbol = :country, year_col::Symbol = :year)::DataFrame
    """Interpolate missing values with gap limits."""
    df = copy(df)
    sort!(df, [group_col, year_col])
    
    if !hasproperty(df, value_col)
        return df
    end
    
    # Initialize flag column if needed
    if !hasproperty(df, :data_quality_flag)
        df[!, :data_quality_flag] = fill(FLAGS.ORIGINAL, nrow(df))
    end
    
    # Process each group
    for group_val in unique(df[!, group_col])
        group_mask = df[!, group_col] .== group_val
        group_data = df[group_mask, :]
        
        if nrow(group_data) < interpolator.min_years
            log_warn("Group $group_val has < $(interpolator.min_years) years, skipping interpolation")
            continue
        end
        
        values = group_data[!, value_col]
        years = group_data[!, year_col]
        
        # Identify and interpolate gaps
        for i in 1:(length(values)-1)
            if ismissing(values[i]) || ismissing(values[i+1])
                continue
            end
            
            gap_size = years[i+1] - years[i] - 1
            
            if gap_size > 0 && gap_size <= interpolator.max_gap
                start_val = values[i]
                end_val = values[i+1]
                
                # Linear interpolation
                for (j, gap_year) in enumerate((years[i]+1):(years[i+1]-1))
                    weight = j / (gap_size + 1)
                    interpolated_val = start_val + weight * (end_val - start_val)
                    
                    # Find row index for this year
                    year_mask = (df[!, group_col] .== group_val) .& (df[!, year_col] .== gap_year)
                    if any(year_mask)
                        idx = findfirst(year_mask)
                        df[idx, value_col] = interpolated_val
                        df[idx, :data_quality_flag] = FLAGS.INTERPOLATED
                        log_debug("Interpolated $group_val year $gap_year: $(@sprintf("%.2f", interpolated_val)) TWh")
                    end
                end
            end
        end
    end
    
    return df
end


function load_config(config_path::String = "config.yaml")::Dict
    """Load configuration from YAML file."""
    if isfile(config_path)
        config = YAML.load_file(config_path)
        log_info("Loaded configuration from $config_path")
        return config
    else
        log_warn("Config file $config_path not found, using defaults")
        return Dict()
    end
end


function load_data(file_path::String, encoding::String = "utf-8")::DataFrame
    """Load data with encoding handling."""
    log_info("Loading data from $file_path")
    
    try
        if endswith(file_path, ".csv")
            df = CSV.read(file_path, DataFrame)
            log_info("Successfully loaded CSV file")
        elseif endswith(file_path, ".xlsx") || endswith(file_path, ".xls")
            error("Excel files not directly supported. Please convert to CSV or use XLSX.jl")
        else
            error("Unsupported file format: $file_path")
        end
    catch e
        log_error("Error loading data: $e")
        rethrow(e)
    end
    
    log_info("Loaded $(nrow(df)) rows, $(ncol(df)) columns")
    
    # Check for duplicates
    duplicates = nonunique(df)
    if any(duplicates)
        log_warn("Found $(sum(duplicates)) duplicate rows")
        df = unique(df)
    end
    
    return df
end


function standardize_columns(df::DataFrame)::DataFrame
    """Standardize column names preserving meaning."""
    log_info("Standardizing column names")
    
    df = copy(df)
    
    # Convert to lowercase first, then clean
    new_names = String[]
    for name in names(df)
        new_name = lowercase(string(name))
        new_name = replace(new_name, ' ' => '_', '-' => '_')
        new_name = replace(new_name, r"[^a-z0-9_]" => "")
        push!(new_names, new_name)
    end
    
    rename!(df, Symbol.(names(df)) .=> Symbol.(new_names))
    
    # Explicit mappings
    column_mapping = Dict(
        "country" => "country",
        "year" => "year",
        "primary_energy" => "primary_energy",
        "primary_energy_consumption" => "primary_energy",
        "energy_consumption" => "primary_energy",
        "electricity_generation" => "electricity_generation",
        "electricity" => "electricity_generation",
        "total_energy" => "primary_energy"
    )
    
    for (old_name, new_name) in column_mapping
        if hasproperty(df, Symbol(old_name)) && !hasproperty(df, Symbol(new_name))
            rename!(df, Symbol(old_name) => Symbol(new_name))
        end
    end
    
    log_info("Standardized columns: $(names(df))")
    return df
end


function load_population_data(file_path::Union{String, Nothing}, config::Dict)::Union{DataFrame, Nothing}
    """Load population data with validation."""
    if file_path !== nothing && isfile(file_path)
        log_info("Loading population data from $file_path")
        pop_df = CSV.read(file_path, DataFrame)
        # Standardize column names
        new_names = [lowercase(replace(string(name), ' ' => '_')) for name in names(pop_df)]
        rename!(pop_df, Symbol.(names(pop_df)) .=> Symbol.(new_names))
        
        # Validate required columns
        required = get(get(config, "population", Dict()), "required_columns", ["country", "year", "population"])
        missing = [col for col in required if !hasproperty(pop_df, Symbol(col))]
        if !isempty(missing)
            log_error("Population file missing required columns: $missing")
            return nothing
        end
        
        return pop_df
    else
        behavior = get(get(config, "population", Dict()), "missing_population_behavior", "warn")
        if behavior == "error"
            error("Population file required but not provided")
        elseif behavior == "warn"
            log_warn("Population file not provided - per-capita calculations will be skipped")
        end
        return nothing
    end
end


function main()
    """Main execution with full validation."""
    # Load configuration
    config = load_config()
    
    # Configuration
    INPUT_FILE = "raw_energy_data.csv"
    POPULATION_FILE = nothing
    GDP_FILE = nothing
    
    # Create audit log
    audit_log = Dict(
        "timestamp" => string(now()),
        "input_file" => INPUT_FILE,
        "config" => config,
        "steps" => []
    )
    
    try
        # Step 1: Load data
        println("=" ^ 80)
        log_info("STEP 1: DATA LOADING")
        println("=" ^ 80)
        df_raw = load_data(INPUT_FILE)
        push!(audit_log["steps"], Dict("step" => "load_data", "rows" => nrow(df_raw), "columns" => ncol(df_raw)))
        
        # Step 2: Schema validation
        println("=" ^ 80)
        log_info("STEP 2: SCHEMA VALIDATION")
        println("=" ^ 80)
        validator = SchemaValidator(config)
        is_valid, errors, warnings = validate(validator, df_raw)
        
        for error in errors
            log_error("SCHEMA ERROR: $error")
        end
        for warning in warnings
            log_warn("SCHEMA WARNING: $warning")
        end
        
        if !is_valid
            log_error("Schema validation failed. Please fix errors before proceeding.")
            return
        end
        
        push!(audit_log["steps"], Dict(
            "step" => "schema_validation",
            "valid" => is_valid,
            "errors" => errors,
            "warnings" => warnings
        ))
        
        # Step 3: Standardize columns
        println("=" ^ 80)
        log_info("STEP 3: COLUMN STANDARDIZATION")
        println("=" ^ 80)
        df_std = standardize_columns(df_raw)
        
        # Step 4: Unit conversion (EXPLICIT ONLY)
        println("=" ^ 80)
        log_info("STEP 4: UNIT CONVERSION (EXPLICIT)")
        println("=" ^ 80)
        converter = UnitConverter(config)
        
        # Find energy columns
        exclude_cols = ["country", "year", "data_quality_flag", "region"]
        energy_cols = [Symbol(col) for col in names(df_std) 
                       if col ∉ exclude_cols && 
                          eltype(df_std[!, Symbol(col)]) <: Union{Real, Missing}]
        
        if isempty(energy_cols)
            log_error("No energy columns found!")
            return
        end
        
        # Convert to TWh
        df_std[!, :total_energy_twh] = create_unified_column(converter, df_std, energy_cols)
        
        # Step 5: Outlier detection
        println("=" ^ 80)
        log_info("STEP 5: OUTLIER DETECTION")
        println("=" ^ 80)
        detector = RobustOutlierDetector(config)
        outlier_mask = detect(detector, df_std[!, :total_energy_twh], 
                             hasproperty(df_std, :country) ? df_std[!, :country] : nothing)
        
        outlier_count = sum(outlier_mask)
        log_info("Detected $outlier_count outliers using $(detector.method) method")
        
        if !hasproperty(df_std, :data_quality_flag)
            df_std[!, :data_quality_flag] = fill(FLAGS.ORIGINAL, nrow(df_std))
        end
        
        df_std[outlier_mask, :data_quality_flag] .= FLAGS.REMOVED_OUTLIER
        df_std[outlier_mask, :total_energy_twh] .= missing
        
        # Step 6: Safe interpolation
        println("=" ^ 80)
        log_info("STEP 6: SAFE INTERPOLATION")
        println("=" ^ 80)
        interpolator = SafeInterpolator(config)
        df_interpolated = interpolate(interpolator, df_std, :total_energy_twh)
        
        # Step 7: Population normalization
        println("=" ^ 80)
        log_info("STEP 7: POPULATION NORMALIZATION")
        println("=" ^ 80)
        pop_df = load_population_data(POPULATION_FILE, config)
        if pop_df !== nothing
            df_interpolated = leftjoin(df_interpolated, pop_df[!, [:country, :year, :population]], on = [:country, :year])
            df_interpolated[!, :energy_per_capita_twh] = df_interpolated[!, :total_energy_twh] ./ df_interpolated[!, :population]
            df_interpolated[!, :energy_per_capita_twh] = [isfinite(x) ? x : missing for x in df_interpolated[!, :energy_per_capita_twh]]
            log_info("Added energy_per_capita_twh column")
        else
            log_warn("Skipping per-capita calculations (no population data)")
        end
        
        # Step 8: Export
        println("=" ^ 80)
        log_info("STEP 8: EXPORT")
        println("=" ^ 80)
        mkpath("output_robust")
        CSV.write("output_robust/cleaned_energy_data.csv", df_interpolated)
        
        # Export audit log
        open("output_robust/audit_log.json", "w") do f
            JSON.print(f, audit_log, 2)
        end
        
        log_info("Processing complete!")
        log_info("Results saved to output_robust/")
        
    catch e
        log_error("Fatal error: $e")
        rethrow(e)
    end
end


# Run main if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

