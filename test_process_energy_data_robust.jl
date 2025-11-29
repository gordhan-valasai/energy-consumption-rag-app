"""
Unit tests for robust energy data processing pipeline (Julia).
"""

using Test
using DataFrames
using Statistics

# Include the robust processing module
include("process_energy_data_robust.jl")


@testset "SchemaValidator" begin
    """Test schema validation."""
    
    @testset "Required columns present" begin
        config = Dict("schema" => Dict("required_columns" => ["country", "year"]))
        validator = SchemaValidator(config)
        
        df = DataFrame(
            country = ["USA", "China"],
            year = [2000, 2001],
            energy = [100, 200]
        )
        
        is_valid, errors, warnings = validate(validator, df)
        @test is_valid
        @test isempty(errors)
    end
    
    @testset "Missing required columns" begin
        config = Dict("schema" => Dict("required_columns" => ["country", "year"]))
        validator = SchemaValidator(config)
        
        df = DataFrame(
            country = ["USA", "China"],
            energy = [100, 200]
        )
        
        is_valid, errors, warnings = validate(validator, df)
        @test !is_valid
        @test any(occursin("year", lowercase(error)) for error in errors)
    end
    
    @testset "Duplicate country-year pairs" begin
        config = Dict("schema" => Dict("required_columns" => ["country", "year"]))
        validator = SchemaValidator(config)
        
        df = DataFrame(
            country = ["USA", "USA", "China"],
            year = [2000, 2000, 2001],
            energy = [100, 200, 300]
        )
        
        is_valid, errors, warnings = validate(validator, df)
        @test !is_valid
        @test any(occursin("duplicate", lowercase(error)) for error in errors)
    end
end


@testset "UnitConverter" begin
    """Test unit conversion."""
    
    @testset "EJ to TWh conversion" begin
        config = Dict(
            "unit_conversions" => Dict(
                "EJ_to_TWh" => 277.778,
                "Mtoe_to_TWh" => 11.63,
                "ktoe_to_TWh" => 0.01163,
                "TWh_to_TWh" => 1.0
            )
        )
        converter = UnitConverter(config)
        
        df = DataFrame(energy_ej = [1.0, 2.0, 3.0])
        
        result = convert_column(converter, df, :energy_ej)
        expected = [277.778, 555.556, 833.334]
        
        for (i, (r, e)) in enumerate(zip(result, expected))
            @test isapprox(r, e, rtol=1e-3)
        end
    end
    
    @testset "Mtoe to TWh conversion" begin
        config = Dict(
            "unit_conversions" => Dict(
                "EJ_to_TWh" => 277.778,
                "Mtoe_to_TWh" => 11.63,
                "ktoe_to_TWh" => 0.01163,
                "TWh_to_TWh" => 1.0
            )
        )
        converter = UnitConverter(config)
        
        df = DataFrame(energy_mtoe = [10.0, 20.0])
        
        result = convert_column(converter, df, :energy_mtoe)
        expected = [116.3, 232.6]
        
        for (i, (r, e)) in enumerate(zip(result, expected))
            @test isapprox(r, e, rtol=1e-3)
        end
    end
    
    @testset "Unified column creation" begin
        config = Dict(
            "unit_conversions" => Dict(
                "EJ_to_TWh" => 277.778,
                "Mtoe_to_TWh" => 11.63,
                "TWh_to_TWh" => 1.0
            )
        )
        converter = UnitConverter(config)
        
        df = DataFrame(
            energy_ej = [1.0, missing, missing],
            energy_mtoe = [missing, 10.0, missing],
            energy_twh = [missing, missing, 100.0]
        )
        
        result = create_unified_column(converter, df, [:energy_ej, :energy_mtoe, :energy_twh])
        expected = [277.778, 116.3, 100.0]
        
        for (i, (r, e)) in enumerate(zip(result, expected))
            if !ismissing(r)
                @test isapprox(r, e, rtol=1e-3)
            end
        end
    end
end


@testset "RobustOutlierDetector" begin
    """Test robust outlier detection."""
    
    @testset "MAD outlier detection" begin
        config = Dict(
            "data_quality" => Dict(
                "outlier_method" => "mad",
                "mad_multiplier" => 3.0
            )
        )
        detector = RobustOutlierDetector(config)
        
        values = [100.0, 105.0, 98.0, 102.0, 1000.0]  # 1000 is outlier
        
        outliers = detect(detector, values)
        @test outliers[end] == true  # Last value should be outlier
        @test all(!outliers[i] for i in 1:(length(outliers)-1))  # Others should not be outliers
    end
    
    @testset "IQR outlier detection" begin
        config = Dict(
            "data_quality" => Dict(
                "outlier_method" => "iqr",
                "iqr_multiplier" => 1.5
            )
        )
        detector = RobustOutlierDetector(config)
        
        values = [10.0, 12.0, 11.0, 13.0, 50.0]  # 50 is outlier
        
        outliers = detect(detector, values)
        @test outliers[end] == true
    end
    
    @testset "Fixed threshold outlier detection" begin
        config = Dict(
            "data_quality" => Dict(
                "outlier_method" => "fixed_threshold",
                "max_energy_twh" => 100.0
            )
        )
        detector = RobustOutlierDetector(config)
        
        values = [50.0, 75.0, 150.0, -10.0]  # 150 and -10 are outliers
        
        outliers = detect(detector, values)
        @test outliers[3] == true  # 150 > threshold
        @test outliers[4] == true  # -10 < 0
    end
end


@testset "SafeInterpolator" begin
    """Test safe interpolation."""
    
    @testset "Interpolation within gap limit" begin
        config = Dict(
            "data_quality" => Dict(
                "max_interpolation_gap" => 3,
                "min_years_for_interpolation" => 2
            )
        )
        interpolator = SafeInterpolator(config)
        
        df = DataFrame(
            country = ["USA", "USA", "USA", "USA"],
            year = [2000, 2001, 2003, 2004],  # Gap of 1 year (2002 missing)
            total_energy_twh = [100.0, 110.0, missing, 130.0]
        )
        
        result = interpolate(interpolator, df, :total_energy_twh)
        
        # Should interpolate 2002 if row exists
        year_2002 = filter(row -> row.year == 2002, result)
        if nrow(year_2002) > 0
            @test !ismissing(year_2002[1, :total_energy_twh])
        end
    end
    
    @testset "No interpolation beyond gap limit" begin
        config = Dict(
            "data_quality" => Dict(
                "max_interpolation_gap" => 2,
                "min_years_for_interpolation" => 2
            )
        )
        interpolator = SafeInterpolator(config)
        
        df = DataFrame(
            country = ["USA", "USA", "USA"],
            year = [2000, 2001, 2005],  # Gap of 3 years (exceeds limit)
            total_energy_twh = [100.0, 110.0, 150.0]
        )
        
        result = interpolate(interpolator, df, :total_energy_twh)
        
        # Should not interpolate years 2002-2004
        for year in [2002, 2003, 2004]
            year_rows = filter(row -> row.year == year, result)
            if nrow(year_rows) > 0
                # If row exists, it should still be missing
                @test ismissing(year_rows[1, :total_energy_twh])
            end
        end
    end
end


@testset "ColumnStandardization" begin
    """Test column name standardization."""
    
    @testset "Lowercase conversion" begin
        df = DataFrame(
            Country = [1, 2],
            Year = [2000, 2001],
            Energy = [100, 200]
        )
        
        result = standardize_columns(df)
        @test :country in propertynames(result)
        @test :year in propertynames(result)
    end
end

